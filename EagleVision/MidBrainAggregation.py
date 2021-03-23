from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import cv2
import PIL

''' 2020_PoolNet
'''

class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, need_x2 =False, need_fuse=False):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2,4,8]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)

    def forward(self, x, x2=None, x3=None):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3))
        return resl

# TODO: 尝试将FAM用于monodepth解码器中，替换多尺度的损失，用来减少混合重影
class EagleFAM(nn.Module):
    def __init__(self, in_chnls,out_chnls):
        super(EagleFAM, self).__init__()

        self.pools_sizes = [2,4,8]
        # 降采样池化
        downPoolLayers,convLayers = [],[]
        for s in self.pools_sizes:
            downPoolLayers.append(nn.AvgPool2d(kernel_size=s,stride=s))
            convLayers.append(nn.Conv2d(in_channels=in_chnls,out_channels=in_chnls,kernel_size=(3,3), stride=1, padding=1, bias=False))

        self.pools = nn.ModuleList(downPoolLayers)
        self.convs = nn.ModuleList(convLayers)
        self.relu = nn.ReLU()

        self.last_conv = nn.Conv2d(in_channels=in_chnls,out_channels=out_chnls,kernel_size=(3, 3), stride=1,padding=1,bias=False)

    def forward(self, input_):
        in_size = input_.size()
        out = input_

        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](input_))
            out = torch.add(out, F.interpolate(y, in_size[2:], mode='bilinear',align_corners=True))
        out = self.relu(out)
        out = self.last_conv(out)
        return out

''' 2019_PFA
'''
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size - 1) // 2  # Padding on one side for stride 1

        self.grp1_conv1k = nn.Conv2d(self.in_channels, self.in_channels // 2, (1, self.kernel_size), padding=(0, pad))
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels // 2, 1, (self.kernel_size, 1), padding=(pad, 0))
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(self.in_channels, self.in_channels // 2, (self.kernel_size, 1), padding=(pad, 0))
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp2_conv1k = nn.Conv2d(self.in_channels // 2, 1, (1, self.kernel_size), padding=(0, pad))
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelwiseAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels // 4)
        self.linear_2 = nn.Linear(self.in_channels // 4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))

        # Activity regularizer
        ca_act_reg = torch.mean(feats)

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()

        return feats, ca_act_reg


''' 2020_DFNet
'''

class CA_Block(nn.Module):
    def __init__(self, in_channels):
        super(CA_Block, self).__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels // 8,kernel_size=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels // 8,out_channels=self.in_channels,kernel_size=1,bias=False)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()
        size = n_b*n_c
        input = input_

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c,1,1))
        # print(feats.shape)
        feats = F.relu(self.conv1(feats))
        feats = torch.sigmoid(self.conv2(feats))

        # Activity regularizer
        weight_vector = torch.mean(feats)

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()

        out = torch.mul(input,weight_vector)

        return out


class AMI_Block(nn.Module):
    def __init__(self, in_chnls,out_chnls):
        super(AMI_Block, self).__init__()

        self.ca = CA_Block(in_chnls)
        self.conv = torch.nn.Conv2d(in_channels=in_chnls,out_channels=out_chnls,kernel_size=3,stride=1)
        self.bn = torch.nn.BatchNorm2d(out_chnls)

    def forward(self, x1,x2):
        x = torch.cat([x1,x2],dim=1)
        y = self.ca(x)
        out = F.relu(self.bn(self.conv(y)))
        return out


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class BiConv2(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5):
        super(BiConv2, self).__init__()

        self.conv_1k = nn.Conv2d(in_channels,out_channels,1)

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size - 1) // 2  # Padding on one side for stride 1

        self.conv1k = nn.Conv2d(self.in_channels, 1, (1, self.kernel_size), padding=(0, pad))
        self.bn1 = nn.BatchNorm2d(1)
        self.convk1 = nn.Conv2d(1, out_channels, (self.kernel_size, 1), padding=(pad, 0))
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input_):
        # Generate Group 1 Features
        feats = self.conv1k(input_)
        feats = F.relu(self.bn1(feats))
        feats = self.convk1(feats)
        feats = F.relu(self.bn2(feats))

        return feats


class MAG_Block(nn.Module):
    def __init__(self,in_chnls,out_chnls):
        super(MAG_Block,self).__init__()

        self.conv_1_1 = nn.Conv2d(in_chnls,out_chnls,kernel_size=1,stride=1,padding=0)
        self.bn1= nn.BatchNorm2d(out_chnls)

        self.conv_3_3 = nn.Conv2d(in_chnls,out_chnls,kernel_size=3,stride=1,padding=1)
        self.bn3= nn.BatchNorm2d(out_chnls)

        # for sep_conv2: padding=(kernel_size-1)//2
        self.sep_conv_5_5 = SeparableConv2d(in_chnls,out_chnls,kernel_size=3,stride=1,padding=2,dilation=2)
        self.bi_conv_5_5 = BiConv2(in_chnls,out_chnls,kernel_size=5)
        self.bn5= nn.BatchNorm2d(out_chnls)

        self.sep_conv_7_7 = SeparableConv2d(in_chnls,out_chnls,kernel_size=3,stride=1,padding=3,dilation=3)
        self.bi_conv_7_7 = BiConv2(in_chnls,out_chnls,kernel_size=7)
        self.bn7= nn.BatchNorm2d(out_chnls)

        self.sep_conv_9_9 = SeparableConv2d(in_chnls,out_chnls,kernel_size=3,stride=1,padding=4,dilation=4)
        self.bi_conv_9_9 = BiConv2(in_chnls,out_chnls,kernel_size=9)
        self.bn9 = nn.BatchNorm2d(out_chnls)

        self.sep_conv_11_11 = SeparableConv2d(in_chnls,out_chnls,kernel_size=3,stride=1,padding=5,dilation=5)
        self.bi_conv_11_11 = BiConv2(in_chnls,out_chnls,kernel_size=11)
        self.bn11= nn.BatchNorm2d(out_chnls)

        self.ca = CA_Block(out_chnls)

    def forward(self, x):
        # x = x.view(128,64,1,1)

        chn_out = []

        f1 = F.relu(self.bn1(self.conv_1_1(x)))
        chn_out.append(f1)

        f2 = F.relu(self.bn3(self.conv_3_3(x)))
        chn_out.append(f2)

        f3l = self.sep_conv_5_5(x)
        f3r = self.bi_conv_5_5(x)
        f3 = torch.add(f3l,f3r)
        f3 = F.relu(self.bn5(f3))
        chn_out.append(f3)

        f4 = torch.add(self.sep_conv_7_7(x),self.bi_conv_7_7(x))
        f4 = F.relu(self.bn7(f4))
        chn_out.append(f4)

        f5 = torch.add(self.sep_conv_9_9(x),self.bi_conv_9_9(x))
        f5 = F.relu(self.bn9(f5))
        chn_out.append(f3)

        f6 = torch.add(self.sep_conv_11_11(x),self.bi_conv_11_11(x))
        f6 = F.relu(self.bn11(f6))
        chn_out.append(f6)

        cat_out = torch.cat(chn_out)
        out = self.ca(cat_out)
        return out


def magTest():
    dummy_input = torch.randn(2,64,28,28)
    # dev = torch.device("cuda" if torch.cuda.Is_availabe() else "cpu")
    mag = MAG_Block(64,128)
    # summary(mag,input_size=(64,64,1,1),batch_size=1,device="cpu")

    print(mag)
    out = mag(dummy_input)
    print('mag output size :', out.size())


def famTest():
    dummy_input = torch.randn(2,4,28,28)
    fam = EagleFAM(4,8)
    fam_out = fam(dummy_input)
    print('fam_out output size :', fam_out.size())
    # summary(fam,input_size=(2,4,28,28),batch_size=1,device="cpu")

def test():
    # dummy_input = torch.randn(2, 4, 64, 64)
    # print('Input Size :', dummy_input.size())
    #
    # # Test Spatial Attention
    # sa = SpatialAttention(4)
    # sa_out = sa(dummy_input)
    # print('Spatial Attention output size :', sa_out.size())
    #
    # # Test Channel-wise Attention
    # ca = ChannelwiseAttention(4)
    # ca_out, reg_val = ca(dummy_input)
    # print('Channel-wise Attention output size :', ca_out.size())
    # print('Channel-wise Attention Regularization value :', reg_val)
    #
    # # Test Channel-wise Attention
    # ca2 = CA_Block(8)
    # dummy_input2 = torch.randn(2, 8, 16, 8)
    # out2 = ca2(dummy_input2)
    # print('ca2 output size :', out2.size())

    # AIM block
    ami = AMI_Block(in_chnls=8,out_chnls=8)
    dummy_input3 = torch.randn(2,4,16,16)
    dummy_input4 = torch.randn(2,4,16,16)
    aim_out = ami(dummy_input3,dummy_input4)
    print('aim_out output size :', aim_out.size())

if __name__ == '__main__':
    # magTest()
    # test()
    famTest()
