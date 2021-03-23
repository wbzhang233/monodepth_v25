# Copyright wbzhang 2021.


from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchsummary import summary
from EagleVision.MidBrainAggregation import ChannelwiseAttention,SpatialAttention

class VggEncode(nn.Module):
    def __init__(self,num_layers,pretrained, num_input_images=1):
        super(VggEncode, self).__init__()

        vgg = {16: models.vgg16,
                   17: models.vgg16_bn,
                   19: models.vgg19,
                   21: models.vgg19_bn}

        if num_layers not in vgg:
            raise ValueError("{} is not a valid number of vgg layers".format(num_layers))

        self.encoder = vgg[num_layers](pretrained)




    def forward(self, input_):
        n_c,n_c,h,w = input_.size()

