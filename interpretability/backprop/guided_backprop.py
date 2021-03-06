"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import LeakyReLU

__all__ = ['GuidedBackprop']


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


class GuidedBackprop:

    def __init__(self, params, device):
        self.model = params.network.to(device)
        self.gradients = None
        self.forward_relu_outputs = []
        torch.autograd.set_detect_anomaly(True)
        self.model.eval()
        self.update_relus()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.layers._modules.items())[0][1][0].layers[0]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for mod in self.model.modules():
            if isinstance(mod, LeakyReLU):
                mod = LeakyReLU(inplace=False, negative_slope=0.1)
                mod.register_backward_hook(relu_backward_hook_function)
                mod.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, params, img_tf, device, detections, class_flag=True, box_flag=True):
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

        gt = torch.empty((detections.shape[0], 4), requires_grad=False)
        gt[:, 2] = torch.from_numpy(detections.width.values) / stride
        gt[:, 3] = torch.from_numpy(detections.height.values) / stride
        gt[:, 0] = torch.from_numpy(detections.x_top_left.values).float() / stride + (gt[:, 2] / 2)
        gt[:, 1] = torch.from_numpy(detections.y_top_left.values).float() / stride + (gt[:, 3] / 2)

        # Find best anchor for each gt
        iou_gt_anchors = bbox_wh_ious(gt, anchors_new)
        _, best_anchors = iou_gt_anchors.max(1)

        gradients_as_arr = []

        for idx, entry in detections.iterrows():
            # Variable
            cls_index = params.class_label_map.index(entry.class_label)

            # Compute middle
            center_x = entry.x_top_left + entry.width/2
            center_y = entry.y_top_left + entry.height/2
            grid_x = int(center_x * width / img_tf.size(2))
            grid_y = int(center_y * height / img_tf.size(3))

            # Choose best anchor
            if entry.status is 'FN':
                anchor_max_idx = best_anchors[idx].item()
            else:
                anchor_max_idx = int(entry.anchor_box)

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
            gradients_as_arr.append(self.gradients.data.cpu().numpy()[0])

        return gradients_as_arr
