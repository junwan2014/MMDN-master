#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 18:44:00 2018

@author: xiang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from MPNCOV.python import MPNCOV


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        planes = int(out_planes/2)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Upsample,self).__init__()
        self.upsample = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        return self.upsample(x)

class HourGlass(nn.Module):
    def __init__(self, depth, num_features):
        super(HourGlass, self).__init__()
        self.depth = depth
        self.features = num_features
        self.Upsample = Upsample(256,256)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = self.Upsample(low3)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


# class MSC_module(nn.Module):
#     """ Self attention Layer"""
#
#     def __init__(self, in_dim, activation):
#         super(MSC_module, self).__init__()
#         self.chanel_in = in_dim
#         self.activation = activation
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)  #
#
#     def forward(self, x_in, x_out):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         m_batchsize, C, width, height = x_out.size()
#         proj_query = self.query_conv(x_out).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
#         proj_key = self.key_conv(x_out).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
#         energy = torch.bmm(proj_query, proj_key)  # transpose check
#         attention = self.softmax(energy)  # BX (N) X (N)
#         proj_value = self.value_conv(x_out).view(m_batchsize, -1, width * height)  # B X C X N
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)
#
#         out = self.gamma * out + x_out
#
#         return out

class SOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]

        ## MPN-COV
        cov_mat = MPNCOV.CovpoolLayer(x_sub) # Global Covariance pooling layer
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
        y_cov = self.conv_du(cov_mat_sum)
        return y_cov*x

class FAN(nn.Module):

    def __init__(self, inplanes, outplanes, stacknumber, bn=False):
        super(FAN, self).__init__()
        self.bn = bn
        if bn:
            self.bn = nn.BatchNorm2d(inplanes)

        # Base part
        self.conv1 = nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.stack_IMCG = nn.ModuleList([IMCG_module(4,256,8) for _ in range(stacknumber)])
        self.conv5 = ConvBlock(256,128)
        self.conv6 = conv3x3(128, outplanes)
        self.Upsample = Upsample(128,128)
        self.gamma1 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        
        if self.bn:
            x = self.bn(x)
            
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2,stride=2)
        x = self.conv3(x)
        # residual = x
        for i,l in enumerate(self.stack_IMCG):
            x = l(x)
        x = self.conv5(x)
        x = self.Upsample(x)
        out = self.conv6(x)
        out = torch.sigmoid(out)
        return out

class IMCG_module(nn.Module):
    """ Self attention Layer"""

    def __init__(self, depth, in_dim, reduction):
        super(IMCG_module, self).__init__()
        self.conv4 = HourGlass(depth, in_dim)
        self.msc = MSC_module(in_dim)
        self.mcc = MCC_module(in_dim, reduction)

    def forward(self, x):
        x_out = self.conv4(x)
        x = self.msc(x, x_out)
        x = self.mcc(x)
        return x



class MCC_module(nn.Module):
    def __init__(self, channel, reduction=8):
        super(MCC_module, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(channel)
        self.conv_fcc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,1, padding=0, bias=True),
            nn.Sigmoid()
        )
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        fcc = x
        fcc = self.avg_pool(fcc)
        fcc = self.conv_fcc(fcc)
        fcc = fcc*x


        ## MPN-COV
        cov_mat = MPNCOV.CovpoolLayer(x_sub) # Global Covariance pooling layer
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
        y_cov = self.conv_du(cov_mat_sum)
        scc = y_cov*x
        mcc = F.relu(self.bn(fcc + scc))
        mcc = mcc + x
        return mcc

class MSC_module(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(MSC_module, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        self.bn = nn.BatchNorm2d(in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x_in, x_out):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x_in.size()
        P = x_in.view(m_batchsize, -1, width * height)
        Q = x_out.view(m_batchsize, -1, width * height)
        PPT = torch.bmm(P.transpose(1, 2), P)
        QQT = torch.bmm(Q.transpose(1, 2), Q)
        PQT = torch.bmm(Q.transpose(1, 2), P)
        Ocaret = self.softmax(PQT + QQT + PPT)
        Qtile = torch.bmm(Q, Ocaret.transpose(1, 2))
        Qtile = Qtile.view(m_batchsize, -1, width, height)
        out = self.gamma * Qtile
        out = F.relu(self.bn(out))
        out = out + x_out
        return out