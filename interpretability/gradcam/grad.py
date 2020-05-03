import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.layers[0], target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        y = self.model.layers[1](output)
        z = self.model.layers[2](output)
        output = self.model.layers[3](torch.cat((y, z), 1))
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask, file_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("data/results/cam2/" + file_name + ".jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, params, img_tf, annos, original_image):
        features, output = self.extractor(img_tf.cuda())

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
            out = output.clone()
            out = out.view(batch, num_anchors, -1, height, width)
            out = out[:, :, 4:, :, :]
            #out[:, :, 4, :, :].sigmoid_()

            # Compute gradient
            one_hot = np.zeros_like(out.clone().cpu().detach(), dtype=np.float32)
            one_hot[0, :, 0, grid_y, grid_x] = 1
            one_hot[0, :, cls_index, grid_y, grid_x] = 1
            out[0][~one_hot.astype(bool)] = 0

            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * out)

            # Zero grads
            self.model.zero_grad()

            one_hot.backward(retain_graph=True)

            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

            target = features[-1]
            target = target.cpu().data.numpy()[0, :]

            weights = np.mean(grads_val, axis=(2, 3))[0, :]
            cam = np.zeros(target.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (416, 416))
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)

            show_cam_on_image(original_image, cam, entry.class_label + '_' + str(idx))


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def gradcam2(params, img_tf, original_image, annos, device):
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.

    grad_cam = GradCam(params.network.to(device), target_layer_names=["17_convbatch"])

    grad_cam(params, img_tf, annos, original_image)
    print('Grad cam 2 completed')

    """gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True))
    gb = gb_model(img_tf, index=annos)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite('data/results/gb.jpg', gb)
    cv2.imwrite('data/results/cam_gb.jpg', cam_gb)"""
