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


def detect(params, dataloader, device, in_image, out_image):

    # Preprocess
    img = Image.open(in_image)
    img_tf = ln.data.transform.Letterbox.apply(img, dimension=params.input_dimension)
    img_tf = tf.ToTensor()(img_tf).unsqueeze(0)

    # Run network
    params.network.to(device)
    img_tf.to(device)
    out = params.network(img_tf.cuda())
    test = params.network(img_tf.cuda())

    test = params.test(test)
    print(test)

    # Postprocess
    out = params.post(out)
    out = ln.data.transform.ReverseLetterbox.apply(out, network_size=params.input_dimension, image_size=img.size)

    # Backpropagation
    backprop(params.network, out, test, det_c.image)

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

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        VOCDataset(os.path.join(args.anno, params.test_set), params, False),
        batch_size=params.mini_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=ln.data.brambox_collate,
    )

    # Run detector
    detect(params, dataloader, device, args.image, args.output)
