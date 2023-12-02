import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['UNet']

BN_EPS = 1e-4


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True,
                 is_relu=True, d3=False):
        """ Convolution + Batch Norm + Relu for 2D feature maps
        """
        super(ConvBnRelu, self).__init__()
        if d3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=True)
            self.bn = nn.BatchNorm3d(out_channels, eps=BN_EPS)
        else:                       
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                                  dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS) 
            # self.bn = GroupNorm(16, out_channels)

        self.relu = nn.ReLU(inplace=True) 
        if is_bn is False:
            self.bn = None

        if is_relu is False:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size, is_bn=True, d3=False):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2 
        self.encode = nn.Sequential(
            ConvBnRelu(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                       groups=1, is_bn=is_bn, d3=d3),
            ConvBnRelu(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                       groups=1, is_bn=is_bn, d3=d3)
        )
        if d3:
            self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        else:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.encode(x)
        x_small = self.max_pool(x)
        return x, x_small


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, is_bn=True, d3=False):
        super(StackDecoder, self).__init__()
        self.d3 = d3
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1,
                       stride=1, groups=1, is_bn=is_bn, d3=d3),
            ConvBnRelu(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                       groups=1, is_bn=is_bn, d3=d3)
        )

    def forward(self, x_big, x):
        if self.d3:
            N, C, D, H, W = x_big.size()
            up = F.interpolate(x, size=(D, H, W), align_corners=False, mode='trilinear')
        else:
            N, C, H, W = x_big.size()
            up = F.interpolate(x, size=(H, W), align_corners=False, mode='bilinear')

        res = torch.cat((up, x_big), 1)
        res = self.decode(res)
        return up, res


class UNet(nn.Module):
    """
    This UNet is not like the UNet in paper,
    where is doesn't halve the channels when upsampling the feature map,
    and in center area, it doesn't double the channels.

    More Flexible Unet with different levels
    2D input : [batch_size, C, H, W]
    3D input : [batch_size, C, D, H, W]
    output : same size
    level : the level of unet
    classes : the channel to output
    cache_res : whether cache the whole feature maps in memory(GPU)
    is_bn : whether use batch normalization
    d3 : whether use 3d Unet
    upsample : if assign a shape, it will upsample the output to the given size,
    """

    def __init__(self, in_ch, nclass=1, level=5,
                 cache_res=False, is_bn=True, d3=False,
                 upsample=None):
        super(UNet, self).__init__()
        self.level = level 
        self.is_bn = is_bn 
        self.d3 = d3 
        self.in_ch = in_ch 

        self.cache_res = cache_res 
        self.res = []

        # list for downs and ups
        self.downs = nn.ModuleList() 
        self.ups = nn.ModuleList()
        # center operation
        self.center = None

        self.bottom_channels = self.construct_down_path(self.downs)

        self.center = ConvBnRelu(self.bottom_channels, self.bottom_channels, kernel_size=1, padding=1, stride=1, is_bn=is_bn, d3=d3)

        self.final_channels = self.construct_up_path(self.ups, self.bottom_channels)

        # final classifier
        self.d3 = d3
        if self.d3:
            self.classify = nn.Conv3d(self.final_channels, nclass, kernel_size=1, padding=0, stride=1, bias=True)
        else:
            self.classify = nn.Conv2d(self.final_channels, nclass, kernel_size=1, padding=0, stride=1, bias=True)

        if upsample is not None: 
            self.final_upsample_shape = upsample
        else:
            self.final_upsample_shape = None

    def construct_down_path(self, downs):
        channels = 64
        downs.append(StackEncoder(self.in_ch, channels, kernel_size=3, is_bn=self.is_bn, d3=self.d3)) 
        for i in range(self.level - 1):
            downs.append(StackEncoder(channels, channels * 2, 3, self.is_bn, d3=self.d3))
            channels *= 2
        return channels  

    def construct_up_path(self, ups, channels):
        for i in range(self.level):
            ups.append(StackDecoder(channels, channels, channels // 2, kernel_size=3,
                                         is_bn=self.is_bn, d3=self.d3))
            channels = channels // 2
        return channels

    def do_down(self, downs, x):
        out = x
        outs = []
        # down sample
        for i in range(self.level):
            current_down, out = downs[i](out)
            outs.append(current_down)
        out = self.center(out)
        outs.append(out)
        return outs

    def do_up(self, ups, outs):
        up_outs = []
        out = outs[-1]
        for i in range(self.level):
            _, out = ups[i](outs[self.level - i - 1], out)
            up_outs.append(out)
        return up_outs

    def forward(self, x):
        x_device = x.device
        # x = x.cuda(next(self.parameters()).device) # *-*
        x = x.to(dtype=torch.float32)
        outs = self.do_down(self.downs, x)
        out = self.do_up(self.ups, outs)[-1]

        # out = self.dual(out)
        out = self.classify(out)

        return out.cuda(x_device)

    def extract_features(self, x):
        x_device = x.device
        x = x.cuda(next(self.parameters()).device)

        outs = self.do_down(self.downs, x)[-1] 
        return outs.cuda(x_device)


def test_2d_flexible_unet():
    net = UNet(1, level=5)
    a = torch.Tensor(np.random.randn(1, 1, 128, 128))
    res = net(a)
    print(res.shape)


def test_3d_unet():
    net = UNet(1, level=2, d3=True)
    a = torch.Tensor(np.random.randn(1, 1, 10, 128, 128))
    res = net(a)
    print(res.shape)


