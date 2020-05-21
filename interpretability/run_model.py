import os
from PIL import Image
import torch
from torchvision import transforms as tf
import lightnet as ln
import brambox as bb
import xml.etree.ElementTree as ET

from interpretability.utils.postprocess import *

__all__ = ['transform', 'identify', 'getImage', 'detect']


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


def transform(params, annos, args_anno):
    letterbox = ln.data.transform.Letterbox(dimension=params.input_dimension)

    # Preprocess
    img = Image.open(getImage(args_anno))
    img_tf = letterbox(img)
    annos = letterbox(annos)

    img_tf = tf.ToTensor()(img_tf).unsqueeze(0)
    img_tf.requires_grad = True

    return img_tf, annos


def run_detect(params, args_anno, device, conf_thresh=0.5):
    letterbox = ln.data.transform.Letterbox(dimension=params.input_dimension)

    # Postprocessing
    post_compose = ln.data.transform.Compose([
        GetBoundingBoxesAnchor(len(params.class_label_map), params.network.anchors, conf_thresh),
        NonMaxSuppression(0.5),
        TensorToBramboxAnchor(params.input_dimension, params.class_label_map),
    ])

    # Preprocess
    img = Image.open(getImage(args_anno))
    img_tf = letterbox(img)
    img_tf = tf.ToTensor()(img_tf).unsqueeze(0)
    img_tf.requires_grad = True

    # Run network
    params.network.to(device).eval()
    with torch.no_grad():
        out = params.network(img_tf.to(device))

    # Postprocess
    out = post_compose(out)
    detections = out.copy()
    out = ln.data.transform.ReverseLetterbox.apply(out, network_size=params.input_dimension, image_size=img.size)

    # Draw
    out['label'] = out['class_label'] + ' [' + (out['confidence'] * 100).round(2).astype(str) + '%]'
    img = bb.util.draw_boxes(img, out)

    return img, detections
