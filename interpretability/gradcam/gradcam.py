"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom

from utils import misc_functions


class CamExtractor:
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.layers[0]._modules.items():
            x = module(x)  # Forward
            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # Forward pass on the classifier
        y = self.model.layers[1](x)
        z = self.model.layers[2](x)
        x = self.model.layers[3](torch.cat((y, z), 1))
        return conv_output, x


class GradCam:
    """
        Produces class activation map
    """
    def __init__(self, params, device, target_layer):
        self.model = params.network.to(device)
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, params, img_tf, original_image, annos, device):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 125, 13, 13)
        conv_output, model_output = self.extractor.forward_pass(img_tf.to(device))

        # Tensor
        # Check dimensions
        if model_output.dim() == 3:
            model_output.unsqueeze_(0)

        # Variables
        batch = model_output.size(0)
        height = model_output.size(2)
        width = model_output.size(3)
        anchors = torch.Tensor(self.model.anchors)
        num_anchors = anchors.shape[0]
        conf_thresh = 0.5

        for idx, entry in annos.iterrows():
            # Variable
            cls_index = params.class_label_map.index(entry.class_label)

            # Compute middle
            center_x = entry.x_top_left + entry.width/2
            center_y = entry.y_top_left + entry.height/2
            grid_x = int(center_x * width / img_tf.size(2))
            grid_y = int(center_y * height / img_tf.size(3))

            # Reform output
            out = model_output.clone()
            conv_out = conv_output.clone()
            out = out.view(batch, num_anchors, -1, height, width)
            out = out[:, :, 4:, :, :]
            #out[:, :, 4, :, :].sigmoid_()

            # Compute gradient
            one_hot_output = torch.zeros_like(out)
            one_hot_output[0, :, 0, grid_y, grid_x] = 1
            one_hot_output[0, :, cls_index, grid_y, grid_x] = 1
            out[~one_hot_output.bool()] = 0

            # Zero grads
            self.model.zero_grad()

            # Backward pass with specified target
            out.backward(gradient=one_hot_output, retain_graph=True)
            # Get hooked gradients
            guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
            # Get convolution outputs
            target = conv_out.data.cpu().numpy()[0]
            # Get weights from gradients
            weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
            # Create empty numpy array for cam
            cam = np.ones(target.shape[1:], dtype=np.float32)
            # Multiply each weight with its conv output and then, sum
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]
            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            #cam = np.uint8(Image.fromarray(cam).resize((img_tf.shape[2],
                           #img_tf.shape[3]), Image.ANTIALIAS))/255
            # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
            # supports resizing numpy matrices with antialiasing, however,
            # when I moved the repository to PIL, this option was out of the window.
            # So, in order to use resizing with ANTIALIAS feature of PIL,
            # I briefly convert matrix to PIL image and then back.
            # If there is a more beautiful way, do not hesitate to send a PR.

            # You can also use the code below instead of the code line above, suggested by @ ptschandl
            cam = zoom(cam, np.array(img_tf[0].shape[1:])/np.array(cam.shape))

            save_class_activation_images(original_image, cam, entry.class_label + '_' + str(idx))


def gradcam(params, img_tf, original_image, annos, device):
    # Grad cam
    grad_cam = GradCam(params, device, target_layer='17_convbatch')
    # Generate cam mask
    grad_cam.generate_cam(params, img_tf, original_image, annos, device)
    print('Grad cam completed')
