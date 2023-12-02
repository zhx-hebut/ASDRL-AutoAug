import torch
import torch.nn as nn
from engine.utils.speed import analyze_network_performance
import functools

Norm = nn.BatchNorm3d


class CBRSeq(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, N=2):
        super(CBRSeq, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            Norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.seq(input)


class BottleNeckSeq(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, N=2):
        super(BottleNeckSeq, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels//N, kernel_size=1, stride=1),
            Norm(out_channels//N),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=out_channels//N, out_channels=out_channels//N, kernel_size=kernel_size, stride=stride, padding=padding),
            Norm(out_channels//N),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=out_channels//N, out_channels=out_channels, kernel_size=1),
            Norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.seq(input)


class GroupSeq(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, N=2):
        super(GroupSeq, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, groups=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            Norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            Norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.seq(input)


def test_bottleneck():

    data_gen = functools.partial(torch.randn, 6, 16, 32, 32, 32)

    a = BottleNeckSeq(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

    b = CBRSeq(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

    c = GroupSeq(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

    print('BottleNeck Structure ....')
    analyze_network_performance(a, data_gen, train_time=250, test_time=250)

    print('\nStandard Convolution ....')
    analyze_network_performance(b, data_gen, train_time=250, test_time=250)

    print('\nSeparable Convolution ...')
    analyze_network_performance(c, data_gen, train_time=250, test_time=250)


if __name__ == '__main__':
    test_bottleneck()