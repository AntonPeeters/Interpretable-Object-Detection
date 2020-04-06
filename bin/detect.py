#!/usr/bin/env python
import os
import argparse
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms as tf
import brambox as bb
import lightnet as ln
from dataset import *
from vanilla_backprop import *
import xml.etree.ElementTree as ET


def identify(xml_file):
    root = ET.parse(xml_file).getroot()
    folder = root.find('folder').text
    filename = os.path.splitext(root.find('filename').text)[0]
    return f'{folder}/JPEGImages/{filename}'


def detect(params, annos, device, in_image, out_image):

    # Preprocess
    img = Image.open(in_image)
    img_tf = ln.data.transform.Letterbox.apply(img, dimension=params.input_dimension)
    img_tf = tf.ToTensor()(img_tf).unsqueeze(0)

    # Run network
    params.network.to(device)
    img_tf.to(device)
    out = params.network(img_tf.cuda())
    test = params.network(img_tf.cuda())
    test2 = params.network(img_tf.cuda())

    test2 = params.test(test2)

    # Tensor
    # Check dimensions
    if test.dim() == 3:
        test.unsqueeze_(0)

    # Variables
    batch = test.size(0)
    h = test.size(2)
    w = test.size(3)
    anchors = torch.Tensor(params.network.anchors)
    num_anchors = anchors.shape[0]
    conf_thresh = 0.5

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w).to(device)
    lin_y = torch.linspace(0, h - 1, h).view(h, 1).repeat(1, w).view(h * w).to(device)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1).to(device)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1).to(device)

    network_output = test.view(batch, num_anchors, -1, h * w)  # -1 == 5+num_classes (we can drop feature maps if 1 class)
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

        # Take max detection
        maxmax = torch.max(cls_max)
        thresh = cls_max == maxmax
        maxidx = cls_max_idx[thresh]
    else:
        cls_max = network_output[:, :, 4, :]
        cls_max_idx = torch.zeros_like(cls_max)

    score_thresh = cls_max > conf_thresh
    new_score_thresh = cls_scores > conf_thresh

    print(new_score_thresh.size(), network_output.size())

    #iets = network_output * score_thresh

    # Mask select boxes > conf_thresh
    coords = network_output.transpose(2, 3)[..., 0:4]
    coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
    scores = cls_max[score_thresh]
    idx = cls_max_idx[score_thresh]

    # Get batch numbers of the detections
    batch_num = score_thresh.view(batch, -1)
    nums = torch.arange(1, batch + 1, dtype=torch.uint8, device=batch_num.device)
    batch_num = (batch_num * nums[:, None])[batch_num] - 1

    print(torch.cat([batch_num[:, None].float(), coords, scores[:, None], idx[:, None]], dim=1))

    # Postprocess
    out = params.post(out)
    out = ln.data.transform.ReverseLetterbox.apply(out, network_size=params.input_dimension, image_size=img.size)

    """label = []

    for annoski in annos:
        label.append(annos.class_label)

    # Backpropagation
    for outski, labelski in zip(test, label):
        out_t = torch.FloatTensor(len(params.class_label_map))
        out_t[outski[6].data.cpu().numpy()] = 1
        backprop(params.network, out_t, params.class_label_map.index(labelski), 'data/pandaou_' + annoski.class_label)"""

    # Draw
    out['label'] = out['class_label'] + ' [' + (out['confidence'] * 100).round(2).astype(str) + '%]'
    img = bb.util.draw_boxes(img, out)

    if out_image is not None:
        img.save(out_image)
    else:
        img.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run trained network on an image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('weight', help='Path to weight file')
    parser.add_argument('image', help='Path to image')
    parser.add_argument('-n', '--network', help='network config file', required=True)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-t', '--thresh', help='Detection Threshold', type=float, default=0.5)
    parser.add_argument('-o', '--output', help='Where to save the image', default=None)
    parser.add_argument('-a', '--anno', help='annotation folder', default='./data')
    args = parser.parse_args()
    
    # Parse arguments
    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            print('CUDA enabled')
            device = torch.device('cuda')
        else:
            print('CUDA not available')

    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight.endswith('.state.pt'):
        params.load(args.weight)
    else:
        params.network.load(args.weight)
    params.post[0].conf_thresh = args.thresh    # Overwrite threshold

    annos = bb.io.load('anno_pascalvoc', args.anno, identify)


    """# Dataloader
    dataloader = torch.utils.data.DataLoader(
        VOCDataset(os.path.join(args.anno, params.test_set), params, False),
        batch_size=params.mini_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=ln.data.brambox_collate,
    )"""

    # Run detector
    detect(params, annos, device, args.image, args.output)
