"""
ESRGAN Generator: RRDBNet (Residual-in-Residual Dense Block Network)
Paper: "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
Wang et al., ECCVW 2018
"""

import torch
import torch.nn as nn
import functools


class DenseBlock(nn.Module):
    def __init__(self, num_features=64, growth_channels=32, bias=True):
        super(DenseBlock, self).__init__()
        gc = growth_channels
        nf = num_features

        self.conv1 = nn.Conv2d(nf,           gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc,      gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc,  gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc,  gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc,  nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * 0.2 + x   # residual scaling β=0.2


class RRDB(nn.Module):
    def __init__(self, num_features=64, growth_channels=32):
        super(RRDB, self).__init__()
        self.rdb1 = DenseBlock(num_features, growth_channels)
        self.rdb2 = DenseBlock(num_features, growth_channels)
        self.rdb3 = DenseBlock(num_features, growth_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x   # outer residual scaling


class RRDBNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64,
                 num_blocks=23, growth_channels=32, scale_factor=4):
        super(RRDBNet, self).__init__()
        self.scale_factor = scale_factor

        self.conv_first  = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        rrdb_block       = functools.partial(RRDB, num_features=num_features,
                                             growth_channels=growth_channels)
        self.trunk       = nn.Sequential(*[rrdb_block() for _ in range(num_blocks)])
        self.conv_trunk  = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upsampling  = self._make_upsample(num_features, scale_factor)
        self.conv_hr     = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last   = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.lrelu       = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _make_upsample(self, nf, scale):
        layers = []
        n = 2 if scale == 4 else 1
        for _ in range(n):
            layers += [nn.Conv2d(nf, nf * 4, 3, 1, 1),
                       nn.PixelShuffle(2),
                       nn.LeakyReLU(0.2, inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        feat  = self.conv_first(x)
        trunk = self.conv_trunk(self.trunk(feat))
        feat  = feat + trunk                    # global residual
        feat  = self.upsampling(feat)
        feat  = self.lrelu(self.conv_hr(feat))
        return self.conv_last(feat)