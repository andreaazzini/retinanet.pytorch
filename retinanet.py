import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from resnet import *


def classification_layer_init(tensor, pi=0.01):
    fill_constant = - math.log((1 - pi) / pi)
    if isinstance(tensor, Variable):
        classification_layer_init(tensor.data)
    return tensor.fill_(fill_constant)

def init_conv_weights(layer):
    nn.init.normal(layer.weight.data, std=0.01)
    nn.init.constant(layer.bias.data, val=0)
    return layer

def conv1x1(in_channels, out_channels, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)
    return layer

def conv3x3(in_channels, out_channels, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)
    return layer

def upsample(feature, sample_feature, scale_factor=2):
    h, w = sample_feature.size()[2:]
    return F.upsample(feature, scale_factor=scale_factor)[:, :, :h, :w]


class FeaturePyramid(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid, self).__init__()

        self.resnet = resnet

        self.pyramid_transformation_3 = conv1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1(2048, 256)

        self.pyramid_transformation_6 = conv3x3(2048, 256, padding=1, stride=2)
        self.pyramid_transformation_7 = conv3x3(256, 256, padding=1, stride=2)

        self.upsample_transform_1 = conv3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3(256, 256, padding=1)

    def forward(self, x):
        _, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))

        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)

        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)
        upsampled_feature_5 = upsample(pyramid_feature_5, pyramid_feature_4)
        pyramid_feature_4 = self.upsample_transform_1(torch.add(upsampled_feature_5, pyramid_feature_4))
        
        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)
        upsampled_feature_4 = upsample(pyramid_feature_4, pyramid_feature_3)
        pyramid_feature_3 = self.upsample_transform_2(torch.add(upsampled_feature_4, pyramid_feature_3))

        return pyramid_feature_3, pyramid_feature_4, pyramid_feature_5, pyramid_feature_6, pyramid_feature_7


class SubNet(nn.Module):
    def __init__(self, k, anchors=9, depth=4, activation=F.relu):
        super(SubNet, self).__init__()
        self.anchors = anchors
        self.activation = activation
        self.base = nn.ModuleList([conv3x3(256, 256, padding=1) for _ in range(depth)])
        self.output = nn.Conv2d(256, k * anchors, kernel_size=3, padding=1)
        classification_layer_init(self.output.weight.data)

    def forward(self, x):
        for layer in self.base:
            x = self.activation(layer(x))
        x = self.output(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), x.size(2) * x.size(3) * self.anchors, -1)
        return x


class RetinaNet(nn.Module):
    backbones = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }

    def __init__(self, backbone='resnet101', num_classes=20, pretrained=True):
        super(RetinaNet, self).__init__()
        self.resnet = RetinaNet.backbones[backbone](pretrained=pretrained)
        self.feature_pyramid = FeaturePyramid(self.resnet)
        self.subnet_boxes = SubNet(4)
        self.subnet_classes = SubNet(num_classes + 1)

    def forward(self, x):
        pyramid_features = self.feature_pyramid(x)
        class_predictions = [self.subnet_classes(p) for p in pyramid_features]
        bbox_predictions = [self.subnet_boxes(p) for p in pyramid_features]
        return torch.cat(bbox_predictions, 1), torch.cat(class_predictions, 1)


if __name__ == '__main__':
    net = RetinaNet().cuda()
    x = Variable(torch.rand(1, 3, 864, 1536).cuda())
    for l in net(x):
        print(l.size())

