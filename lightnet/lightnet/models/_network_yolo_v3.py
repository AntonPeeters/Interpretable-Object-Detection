#
#   Darknet YOLOv3 model
#   Copyright EAVISE
#

from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import lightnet.network as lnn

__all__ = ['YoloV3']


class YoloV3(lnn.module.Darknet):
    """ Yolo v3 implementation :cite:`yolo_v3`.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 3D list with anchor values; Default **Yolo v3 anchors**

    Attributes:
        self.stride: Subsampling factors of the network (input dimensions should be a multiple of these numbers)
        self.remap_darknet53: Remapping rules for weights from the `~lightnet.models.Darknet53` model.

    Note:
        Unlike YoloV2, the anchors here are defined as multiples of the input dimensions and not as a multiple of the output dimensions!
        The anchor list also has one more dimension than the one from YoloV2, in order to differentiate which anchors belong to which stride.

    Warning:
        The :class:`~lightnet.network.loss.MultiScaleRegionLoss` and :class:`~lightnet.data.transform.GetMultiScaleBoundingBoxes`
        do not implement the overlapping class labels of the original implementation.
        Your weight files from darknet will thus not have the same accuracies as in darknet itself.
    """
    stride = (32, 16, 8)
    remap_darknet53 = [
        (r'^layers.([a-w]_)',   r'extractor.\1'),   # Residual layers
        (r'^layers.(\d_)',      r'extractor.\1'),   # layers 1, 2, 5
        (r'^layers.([124]\d_)', r'extractor.\1'),   # layers 10, 27, 44
    ]

    def __init__(self, num_classes=20, input_channels=3, anchors=[[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]]):
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable) and not isinstance(anchors[0][0], Iterable):
            raise TypeError('Anchors need to be a 3D list of numbers')

        # Parameters
        self.num_classes = num_classes
        self.anchors = []   # YoloV3 defines anchors as a multiple of the input dimensions of the network as opposed to the output dimensions
        for i, s in enumerate(self.stride):
            self.anchors.append([(a[0] / s, a[1] / s) for a in anchors[i]])

        # Network
        self.extractor = lnn.layer.SelectiveSequential(
            ['k_residual', 's_residual'],
            OrderedDict([
                ('1_convbatch',         lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1)),
                ('2_convbatch',         lnn.layer.Conv2dBatchReLU(32, 64, 3, 2, 1)),
                ('a_residual',          lnn.layer.Residual(OrderedDict([
                    ('3_convbatch',     lnn.layer.Conv2dBatchReLU(64, 32, 1, 1, 0)),
                    ('4_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ]))),
                ('5_convbatch',         lnn.layer.Conv2dBatchReLU(64, 128, 3, 2, 1)),
                ('b_residual',          lnn.layer.Residual(OrderedDict([
                    ('6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                    ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ]))),
                ('c_residual',          lnn.layer.Residual(OrderedDict([
                    ('8_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                    ('9_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ]))),
                ('10_convbatch',        lnn.layer.Conv2dBatchReLU(128, 256, 3, 2, 1)),
                ('d_residual',          lnn.layer.Residual(OrderedDict([
                    ('11_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('12_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('e_residual',          lnn.layer.Residual(OrderedDict([
                    ('13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('14_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('f_residual',          lnn.layer.Residual(OrderedDict([
                    ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('16_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('g_residual',          lnn.layer.Residual(OrderedDict([
                    ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('18_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('h_residual',          lnn.layer.Residual(OrderedDict([
                    ('19_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('20_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('i_residual',          lnn.layer.Residual(OrderedDict([
                    ('21_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('22_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('j_residual',          lnn.layer.Residual(OrderedDict([
                    ('23_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('24_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('k_residual',          lnn.layer.Residual(OrderedDict([
                    ('25_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('26_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('27_convbatch',        lnn.layer.Conv2dBatchReLU(256, 512, 3, 2, 1)),
                ('l_residual',          lnn.layer.Residual(OrderedDict([
                    ('28_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('29_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('m_residual',          lnn.layer.Residual(OrderedDict([
                    ('30_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('31_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('n_residual',          lnn.layer.Residual(OrderedDict([
                    ('32_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('33_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('o_residual',          lnn.layer.Residual(OrderedDict([
                    ('34_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('35_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('p_residual',          lnn.layer.Residual(OrderedDict([
                    ('36_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('37_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('q_residual',          lnn.layer.Residual(OrderedDict([
                    ('38_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('39_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('r_residual',          lnn.layer.Residual(OrderedDict([
                    ('40_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('41_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('s_residual',          lnn.layer.Residual(OrderedDict([
                    ('42_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('43_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('44_convbatch',        lnn.layer.Conv2dBatchReLU(512, 1024, 3, 2, 1)),
                ('t_residual',          lnn.layer.Residual(OrderedDict([
                    ('45_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('46_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('u_residual',          lnn.layer.Residual(OrderedDict([
                    ('47_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('48_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('v_residual',          lnn.layer.Residual(OrderedDict([
                    ('49_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('50_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('w_residual',          lnn.layer.Residual(OrderedDict([
                    ('51_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('52_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
            ]),
        )

        self.detector = nn.ModuleList([
            # Sequence 0 : input = extractor
            lnn.layer.SelectiveSequential(
                ['57_convbatch'],
                OrderedDict([
                    ('53_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('54_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                    ('55_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('56_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                    ('57_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('58_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                    ('59_conv',         nn.Conv2d(1024, len(self.anchors[0])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),

            # Sequence 1 : input = 57_convbatch
            nn.Sequential(
                OrderedDict([
                    ('60_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('61_upsample',     nn.Upsample(scale_factor=2, mode='nearest')),
                ])
            ),

            # Sequence 2 : input = 61_upsample and s_residual
            lnn.layer.SelectiveSequential(
                ['66_convbatch'],
                OrderedDict([
                    ('62_convbatch',    lnn.layer.Conv2dBatchReLU(256+512, 256, 1, 1, 0)),
                    ('63_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('64_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('65_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('66_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('67_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('68_conv',         nn.Conv2d(512, len(self.anchors[1])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),

            # Sequence 3 : input = 66_convbatch
            nn.Sequential(
                OrderedDict([
                    ('69_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('70_upsample',     nn.Upsample(scale_factor=2, mode='nearest')),
                ])
            ),

            # Sequence 4 : input = 70_upsample and k_residual
            nn.Sequential(
                OrderedDict([
                    ('71_convbatch',    lnn.layer.Conv2dBatchReLU(128+256, 128, 1, 1, 0)),
                    ('72_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                    ('73_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('74_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                    ('75_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('76_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                    ('77_conv',         nn.Conv2d(256, len(self.anchors[2])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),
        ])

    def forward(self, x):
        out = [None, None, None]

        # Feature extractor
        x, inter_features = self.extractor(x)

        # detector 0
        out[0], x = self.detector[0](x)

        # detector 1
        x = self.detector[1](x)
        out[1], x = self.detector[2](torch.cat((x, inter_features['s_residual']), 1))

        # detector 2
        x = self.detector[3](x)
        out[2] = self.detector[4](torch.cat((x, inter_features['k_residual']), 1))

        return out
