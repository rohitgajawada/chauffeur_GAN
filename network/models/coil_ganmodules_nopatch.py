import os.path as osp
import torch
import torch.nn as nn
from torch.autograd import Variable

from .model_utils import get_upsampling_weight
from .model_utils import ResnetBlock
from .building_blocks import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join

class _netG(nn.Module):
    def __init__(self, loss='LSGAN', skip=True):
        super(_netG, self).__init__()

        self.p1 = nn.Sequential(Conv(params={'channel_sizes': [3, 32, 32, 64],
                                         'kernel_sizes': [5] + [3]*2,
                                         'strides': [2, 1, 2],
                                         'dropouts': [0.2] + [0.4]*2,
                                         'end_layer': False}))

        self.p2 = nn.Sequential(Conv(params={'channel_sizes': [64, 64, 128],
                                         'kernel_sizes': [3]*2,
                                         'strides': [1, 2],
                                         'dropouts': [0.4]*2,
                                         'end_layer': False}))

        self.p3 = nn.Sequential(Conv(params={'channel_sizes': [128, 128, 256],
                                         'kernel_sizes': [3]*2,
                                         'strides': [1, 1],
                                         'dropouts': [0.4]*2,
                                         'end_layer': False}))

        self.ndim1 = 256
        self.ndim2 = 64
        ngf = 64

        #Removed the batchnorm in the bottleneck layer
        #Added biases on all wherever instance norm is
        #Try relu type = 2 in decoder
        #I have changed norm type to 1, it was 2 initially
        #I have changed all batchnorms to instancenorms except in encoder
        self.stage1_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.ndim1, ngf*4, 7, 1, 0, bias=True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
            ResnetBlock(ngf*4, norm_type=2, bias=True),
            ResnetBlock(ngf*4, norm_type=2, bias=True),
            ResnetBlock(ngf*4, norm_type=2, bias=True)
        )

        self.stage2_upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 2, 1, 0, bias=True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
            ResnetBlock(ngf * 4, norm_type=2, bias=True),
            ResnetBlock(ngf * 4, norm_type=2, bias=True),
            ResnetBlock(ngf * 4, norm_type=2, bias=True)
        )

        self.skip = skip
        if skip:
            self.stage3_upsample = nn.Sequential(
                nn.ConvTranspose2d(ngf*4 + self.ndim2, ngf*2, 5, 2, 0, 0, bias=True),
                nn.InstanceNorm2d(ngf*2),
                nn.ReLU(True),
                ResnetBlock(ngf*2, norm_type=2, bias=True),
                ResnetBlock(ngf*2, norm_type=2, bias=True),
                ResnetBlock(ngf*2, norm_type=2, bias=True)
            )
        else:
            self.stage3_upsample = nn.Sequential(
                nn.ConvTranspose2d(ngf*4, ngf*2, 5, 2, 0, 0, bias=True),
                nn.InstanceNorm2d(ngf*2),
                nn.ReLU(True),
                ResnetBlock(ngf*2, norm_type=2, bias=True),
                ResnetBlock(ngf*2, norm_type=2, bias=True),
            )

        self.stage4_upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, 3, 8, 2),
            nn.Sigmoid()
        )


    def forward(self, x):
        p1 = self.p1(x)
        x = self.p2(p1)
        x = self.p3(x)

        x = self.stage1_upsample(x)
        x = self.stage2_upsample(x)
        if self.skip:
            x = torch.cat((x, p1), 1)
        x = self.stage3_upsample(x)
        x = self.stage4_upsample(x)

        return x

class _netD(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=4, norm_layer=torch.nn.InstanceNorm2d, loss='LSGAN'):
        super(_netD, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = (1, 0)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if loss == 'NORMAL':
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)



# netG = _netG()
# for m in netG.modules():
#     if isinstance(m, nn.Conv2d):
#         print(m)
#         m.register_forward_hook(activ_forward_hook)
#     if isinstance(m, nn.ConvTranspose2d):
#         print(m)
#         m.register_forward_hook(activ_forward_hook)

# x = Variable(torch.randn(1, 3, 88, 200))
# out = netG(x)
# Output size is [1, 3, 88, 200]

def activ_forward_hook(self, inputs, outputs):
    print(outputs.size())
    print("-------------------")

netD = _netD()
for m in netD.modules():
    if isinstance(m, nn.Conv2d):
        print(m)
        m.register_forward_hook(activ_forward_hook)

x = Variable(torch.randn(1, 3, 88, 200))
out = netD(x)
