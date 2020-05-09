import os
from PIL import Image
from torchvision import transforms as tf
import lightnet as ln
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


def detect_new(params, annos, args_anno, device):
    letterbox = ln.data.transform.Letterbox(dimension=params.input_dimension)

    # Preprocess
    img = Image.open(getImage(args_anno))
    img_tf = letterbox(img)
    annos = letterbox(annos)

    img_tf = tf.ToTensor()(img_tf).unsqueeze(0)
    img_tf.requires_grad = True

    # Run network
    params.network.to(device)

    return params, img_tf, annos
