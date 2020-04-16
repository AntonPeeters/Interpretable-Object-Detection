"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch

from misc_functions import convert_to_grayscale, save_gradient_images


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

    def generate_gradients(self, params, img_tf, annos, device):
        output = self.model(img_tf.to(device))

        # Tensor
        # Check dimensions
        if output.dim() == 3:
            output.unsqueeze_(0)

        # Variables
        batch = output.size(0)
        h = output.size(2)
        w = output.size(3)
        anchors = torch.Tensor(self.model.anchors)
        num_anchors = anchors.shape[0]
        conf_thresh = 0.5

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w).to(device)
        lin_y = torch.linspace(0, h - 1, h).view(h, 1).repeat(1, w).view(h * w).to(device)
        anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1).to(device)
        anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1).to(device)

        network_output = output.clone()

        network_output = network_output.view(batch, num_anchors, -1,
                                             h * w)  # -1 == 5+num_classes (we can drop feature maps if 1 class)
        network_output[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)  # X center
        network_output[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)  # Y center
        network_output[:, :, 2, :].exp_().mul_(anchor_w).div_(w)  # Width
        network_output[:, :, 3, :].exp_().mul_(anchor_h).div_(h)  # Height
        network_output[:, :, 4, :].sigmoid_()  # Box score

        # Compute class_score
        if len(params.class_label_map) > 1:
            with torch.no_grad():
                cls_scores = torch.nn.functional.softmax(network_output[:, :, 5:, :], 2)
            cls_max, cls_max_idx = torch.max(cls_scores, 2)
            cls_max_idx = cls_max_idx.float()
            cls_max.mul_(network_output[:, :, 4, :])
        else:
            cls_max = network_output[:, :, 4, :]
            cls_max_idx = torch.zeros_like(cls_max)

        score_thresh = cls_max > conf_thresh

        # Mask select boxes > conf_thresh
        coords = network_output.transpose(2, 3)[..., 0:4]
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = cls_max[score_thresh]
        idx = cls_max_idx[score_thresh]

        # Get batch numbers of the detections
        batch_num = score_thresh.view(batch, -1)
        nums = torch.arange(1, batch + 1, dtype=torch.uint8, device=batch_num.device)
        batch_num = (batch_num * nums[:, None])[batch_num] - 1

        #print(torch.cat([batch_num[:, None].float(), coords, scores[:, None], idx[:, None]], dim=1))

        for idx, entry in annos.iterrows():
            # Variable
            cls_index = params.class_label_map.index(entry.class_label)

            # Compute middle
            center_x = entry.x_top_left + entry.width/2
            center_y = entry.y_top_left + entry.height/2
            grid_x = int(center_x * w / img_tf.size(2))
            grid_y = int(center_y * h / img_tf.size(3))

            # Compute anchor
            cls_scores = torch.nn.functional.softmax(network_output[:, :, cls_index, grid_x + h * grid_y], 1)
            anchor_max, anchor_max_idx = torch.max(cls_scores, 1)

            # Reform output
            out = output.clone()
            out = out.view(batch, num_anchors, -1, h, w)
            out = out[:, :, 5:, :, :]
            out[:, :, 4, :, :].sigmoid()

            # Compute gradient
            one_hot_output = torch.zeros_like(out)
            one_hot_output[0, anchor_max_idx, 0, grid_y, grid_x] = 1
            one_hot_output[0, anchor_max_idx, cls_index, grid_y, grid_x] = 1
            out[~one_hot_output.bool()] = 0

            # Zero grads
            self.model.zero_grad()

            # Backward pass
            out.backward(retain_graph=True, gradient=one_hot_output)

            # Convert Pytorch variable to numpy array
            # [0] to get rid of the first channel (1,3,224,224)
            gradients_as_arr = self.gradients.data.cpu().numpy()[0]

            # Save colored gradients
            save_gradient_images(gradients_as_arr, entry.class_label + '_' + str(idx) + '_color')

            # Convert to grayscale
            grayscale_vanilla_grads = convert_to_grayscale(gradients_as_arr)

            # Save grayscale gradients
            save_gradient_images(grayscale_vanilla_grads, entry.class_label + '_' + str(idx) + '_gray')


def backprop(params, img_tf, annos, device):
    # Vanilla backprop
    VBP = VanillaBackprop(params, device)
    # Generate gradients
    VBP.generate_gradients(params, img_tf, annos, device)
    print('Vanilla backprop completed')
