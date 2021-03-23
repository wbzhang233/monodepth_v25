# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
import EagleVision
from EagleVision.MidBrainAggregation import EagleFAM,DeepPoolLayer


class EagleDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True,use_fam=True):
        super(EagleDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.use_fam = use_fam
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc # [64, 64, 128, 256, 512]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        # TODO:FAM模块
        self.fams = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.fams[("fam", i, 0)] = DeepPoolLayer(num_ch_in,num_ch_in)
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.fams[("fam", i, 1)] = DeepPoolLayer(num_ch_in,num_ch_in)
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)


        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        # self.convlist = list(self.convs.values())
        # self.convs.keys()
        self.decoder = nn.ModuleList(list(self.convs.values())+list(self.fams.values()))
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_features):
        self.outputs = {}

        # TODO: FAM_decoder-->在每次upconv之前使用FAM模块减少重影
        x = input_features[-1]
        for i in range(4, -1, -1):
            # TODO:添加FAM模块
            if self.use_fam:
                x = self.fams[("fam", i, 0)](x)
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            # TODO:添加FAM模块
            if self.use_fam:
                x = self.fams[("fam", i, 1)](x)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

def decoderTest():
    num_ch_enc = np.array([64, 64, 128, 256, 512])
    scales = [0, 1, 2, 3]
    decoder = EagleDepthDecoder(num_ch_enc,scales)
    print("decoder.convs: ",decoder.convs,"\n")
    print("decoder: ",decoder,"\n")



if __name__=="__main__":
    decoderTest()