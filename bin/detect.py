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


def getImage(xml_file):
    root = ET.parse(xml_file).getroot()
    folder = root.find('folder').text
    path = xml_file.split(folder)[0]
    filename = root.find('filename').text
    return path + f'{folder}/JPEGImages/{filename}'


def identify(xml_file):
    root = ET.parse(xml_file).getroot()
    folder = root.find('folder').text
    filename = os.path.splitext(root.find('filename').text)[0]
    return f'{folder}/JPEGImages/{filename}'


def detect(params, annos, image, device, out_image):

    # Preprocess
    img = Image.open(image)
    img_tf = ln.data.transform.Letterbox.apply(img, dimension=params.input_dimension)
    img_tf = tf.ToTensor()(img_tf).unsqueeze(0)
    img_tf.requires_grad = True

    # Run network
    params.network.to(device)
    out = params.network(img_tf.to(device))

    # Backpropagation
    backprop(params, img_tf, annos, device)

    # Postprocess
    out = params.post(out)
    out = ln.data.transform.ReverseLetterbox.apply(out, network_size=params.input_dimension, image_size=img.size)

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
    parser.add_argument('anno', help='Path to image annotation')
    parser.add_argument('-n', '--network', help='network config file', required=True)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-t', '--thresh', help='Detection Threshold', type=float, default=0.5)
    parser.add_argument('-o', '--output', help='Where to save the image', default=None)
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

    image = getImage(args.anno)

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
    detect(params, annos, image, device, args.output)
