import os.path as osp
import torch
import torch.nn as nn
import numpy as np

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class ResnetBlock(nn.Module):
    def __init__(self, feat_dim, norm_type=1, bias=True, relu_type=1):
        super(ResnetBlock, self).__init__()
        self.bias = bias
        self.conv_layer = nn.Conv2d(feat_dim, feat_dim, 3, 1, 1, bias=self.bias)
        self.dropout = nn.Dropout(0.5)
        if relu_type==1:
            self.activation = nn.ReLU(True)
        else:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
            
        if norm_type==1:
            self.normalization_layer = nn.BatchNorm2d(feat_dim)
        else:
            self.normalization_layer = nn.InstanceNorm2d(feat_dim)
            
    def forward(self, x):
        response = self.dropout(self.activation(self.normalization_layer(self.conv_layer(x))))
        out = x + response
        return out

class DilatedResnetBlock(nn.Module):
    def __init__(self, feat_dim, norm_type=1, bias=True, relu_type=1):
        super(DilatedResnetBlock, self).__init__()
        self.bias = bias
        self.conv_layer = nn.Conv2d(feat_dim, feat_dim, 3, 1, 2, dilation=2, bias=self.bias)
        if relu_type==1:
            self.activation = nn.ReLU(True)
        else:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
            
        if norm_type==1:
            self.normalization_layer = nn.BatchNorm2d(feat_dim)
        else:
            self.normalization_layer = nn.InstanceNorm2d(feat_dim)
            
    def forward(self, x):
        response = self.activation(self.normalization_layer(self.conv_layer(x)))
        out = x + response
        return out

