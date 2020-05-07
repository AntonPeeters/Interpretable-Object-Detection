"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch

from interpretability.utils.misc_functions import save_images, normalize

__all__ = ['VanillaBackprop']


def bbox_wh_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Returns:
        torch.Tensor[len(boxes1) X len(boxes2)]: IOU values when discarding X/Y offsets (aka. as if they were zero)

    Note:
        Tensor format: [[xc, yc, w, h],...]
    """
    b1w = boxes1[:, 2].unsqueeze(1)
    b1h = boxes1[:, 3].unsqueeze(1)
    b2w = boxes2[:, 2]
    b2h = boxes2[:, 3]

    intersections = b1w.min(b2w) * b1h.min(b2h)
    unions = (b1w * b1h) + (b2w * b2h) - intersections

    return intersections / unions


class VanillaBackprop:

    def __init__(self, params, device):
        self.model = params.network.to(device)
        self.gradients = None
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.layers._modules.items())[0][1][0].layers[0]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, params, img_tf, annos, device, threshold=1.0, class_flag=True, box_flag=True):
        # Run model
        output = self.model(img_tf.to(device))

        # Tensor
        # Check dimensions
        if output.dim() == 3:
            output.unsqueeze_(0)

        # Variables
        batch = output.size(0)
        height = output.size(2)
        width = output.size(3)
        anchors = torch.Tensor(self.model.anchors)
        num_anchors = anchors.shape[0]
        stride = self.model.stride

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, width - 1, width).repeat(height, 1).view(height * width).to(device)
        lin_y = torch.linspace(0, height - 1, height).view(height, 1).repeat(1, width).view(height * width).to(device)
        anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1).to(device)
        anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1).to(device)

        network_output = output.clone()

        network_output = network_output.view(batch, num_anchors, -1,
                                             height * width)  # -1 == 5+num_classes (we can drop feature maps if 1 class)
        network_output[:, :, 0, :].sigmoid_().add_(lin_x).div_(width)  # X center
        network_output[:, :, 1, :].sigmoid_().add_(lin_y).div_(height)  # Y center
        network_output[:, :, 2, :].exp_().mul_(anchor_w).div_(width)  # Width
        network_output[:, :, 3, :].exp_().mul_(anchor_h).div_(height)  # Height
        network_output[:, :, 4, :].sigmoid_()  # Box score

        # Create ground_truth tensor
        anchors_new = torch.cat([torch.zeros_like(anchors), anchors], 1)

        gt = torch.empty((annos.shape[0], 4), requires_grad=False)
        gt[:, 2] = torch.from_numpy(annos.width.values) / stride
        gt[:, 3] = torch.from_numpy(annos.height.values) / stride
        gt[:, 0] = torch.from_numpy(annos.x_top_left.values).float() / stride + (gt[:, 2] / 2)
        gt[:, 1] = torch.from_numpy(annos.y_top_left.values).float() / stride + (gt[:, 3] / 2)

        # Find best anchor for each gt
        iou_gt_anchors = bbox_wh_ious(gt, anchors_new)
        _, best_anchors = iou_gt_anchors.max(1)

        gradients_as_arr = []
        gradients_as_ten = []

        for idx, entry in annos.iterrows():
            # Variable
            cls_index = params.class_label_map.index(entry.class_label)

            # Compute middle
            center_x = entry.x_top_left + entry.width/2
            center_y = entry.y_top_left + entry.height/2
            grid_x = int(center_x * width / img_tf.size(2))
            grid_y = int(center_y * height / img_tf.size(3))

            # Compute best anchor
            cls_scores = torch.nn.functional.softmax(network_output[:, :, 5 + cls_index, grid_x + grid_y * height], 1)
            cls_scores.mul_(network_output[:, :, 4, grid_x + grid_y * height])
            value, anchor_max_idx = torch.max(cls_scores, 1)
            print("Uit annos:", anchor_max_idx.item(), ", Berekend:", best_anchors[idx].item())
            if value < threshold:
                anchor_max_idx = best_anchors[idx]

            print("Gekozen:", anchor_max_idx.item())

            # Reform output
            out = output.clone()
            out = out.view(batch, num_anchors, -1, height, width)

            # Box only
            if box_flag is True and class_flag is False:
                out = out[:, :, 4, :, :]
                # Compute gradient
                one_hot_output = torch.zeros_like(out)
                one_hot_output[0, anchor_max_idx, grid_y, grid_x] = 1
                out[~one_hot_output.bool()] = 0

            # Class only
            elif box_flag is False and class_flag is True:
                out = out[:, :, 5:, :, :]
                # Compute gradient
                one_hot_output = torch.zeros_like(out)
                one_hot_output[0, anchor_max_idx, cls_index, grid_y, grid_x] = 1
                out[~one_hot_output.bool()] = 0

            # Box and class
            else:
                out = out[:, :, 4:, :, :]
                # Compute gradient
                one_hot_output = torch.zeros_like(out)
                one_hot_output[0, anchor_max_idx, 0, grid_y, grid_x] = 1
                one_hot_output[0, anchor_max_idx, 1 + cls_index, grid_y, grid_x] = 1
                out[~one_hot_output.bool()] = 0

            # Zero grads
            self.model.zero_grad()

            # Backward pass
            out.backward(retain_graph=True, gradient=one_hot_output)

            # Convert Pytorch variable to numpy array
            # [0] to get rid of the first channel (1,3,416,416)
            gradients_as_ten.append(self.gradients[0])
            gradients_as_arr.append(self.gradients.data.cpu().numpy()[0])

        #torch.save(gradients_as_ten, "data/results/raw/tensor" + annos.image[0].split('/')[-1] + ".pt")
        return gradients_as_arr


def backpropagation(params, img_tf, annos, device):
    # Vanilla backprop
    VBP = VanillaBackprop(params, device)
    # Generate gradients
    gradients_as_arr = VBP.generate_gradients(params, img_tf, annos, device)

    # Normalize
    gradients_list = normalize(gradients_as_arr)

    # Save colored gradients
    save_images(gradients_list, '_color')

    # Normalize to grayscale
    gradients_list = normalize(gradients_as_arr, True)

    # Save grayscale gradients
    save_images(gradients_list, '_gray')
    print('Vanilla backprop completed')
