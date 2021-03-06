# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchsummary import summary
from EagleVision.MidBrainAggregation import ChannelwiseAttention,SpatialAttention

# TODO: we don't change this function
class EagleResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(EagleResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# TODO: we don't change this function
def eagle_resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = eagle_resnet_multiimage_input(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class EagleResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(EagleResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        #  channel_wise
        attention_chnls = []
        for num_ch in self.num_ch_enc:
            attention_chnls.append(ChannelwiseAttention(num_ch))

        self.attention_chnls = nn.ModuleList(attention_chnls)


        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = eagle_resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []

        # resnet conv1 feature
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x0 = self.encoder.relu(x)
        f0,ca_act_reg0 = self.attention_chnls[0](x0)
        self.features.append(f0)

        # resnet layer1 feature
        x1 = self.encoder.layer1(self.encoder.maxpool(x0))
        f1,ca_act_reg1 = self.attention_chnls[1](x1)
        self.features.append(f1)

        # resnet layer2 feature
        x2 = self.encoder.layer2(x1)
        f2,ca_act_reg2 = self.attention_chnls[2](x2)
        self.features.append(f2)

        # resnet layer3 feature
        x3 = self.encoder.layer3(x2)
        f3,ca_act_reg3 = self.attention_chnls[3](x3)
        self.features.append(f3)

        # resnet layer4 feature
        x4 = self.encoder.layer4(x3)
        f4,ca_act_reg4 = self.attention_chnls[4](x4)
        self.features.append(f4)

        return self.features


def resnetTest():
    num_layers = 18
    pretrained = "pretrained"
    resnets = {18: models.resnet18,
               34: models.resnet34,
               50: models.resnet50,
               101: models.resnet101,
               152: models.resnet152}

    # encoder = resnets[num_layers](pretrained)
    # print(encoder)
    #
    # print("conv1: ",encoder.conv1,"\n")
    # print("layer1: ",encoder.layer1,"\n")

    encoder1 = EagleResnetEncoder(num_layers,pretrained,1)
    print(encoder1)


if __name__ == '__main__':
    resnetTest()