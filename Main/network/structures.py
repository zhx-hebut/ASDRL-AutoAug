import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, n_feats, att_map=False):
        super(SpatialAttention, self).__init__()
        self.att_map = att_map
        self.n_feats = n_feats
        self.attention = nn.Conv3d(n_feats, 1, kernel_size=1)

    def forward(self, reference, target):
        attention = self.attention(reference).sigmoid()
        # show_graphs([reference[0, 0, 0, ...].cpu(), target[0, 0, 0, ...].cpu(), attention[0, 0, 0, ...].cpu()],
        #             ['reference', 'target', 'attention map'])
        if self.att_map:
            return attention
        else:
            return attention * target + reference


class GAP(nn.Module):
    def __init__(self, in_ch):
        super(GAP, self).__init__()
        self.gap  = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(in_ch, in_ch, kernel_size=1)

    def forward(self, x):
        global_context = self.gap(x)
        return self.conv(x + global_context)


class ASPP(nn.Module):
    def __init__(self, in_ch, dilations=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.dilated_convs = nn.ModuleList()
        for dilation in dilations:
            # pad = (dilation x (kernel-1)) / 2
            self.dilated_convs.append(nn.Conv3d(in_ch, in_ch, kernel_size=3, dilation=dilation, padding=dilation))
        self.gap  = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv1x1 = nn.Conv3d(in_ch*(1+len(dilations)), in_ch, kernel_size=1)

    def forward(self, x):
        global_context = self.gap(x)
        global_context = F.interpolate(global_context, x.size()[2:], align_corners=False, mode='trilinear')
        res = [global_context]
        for conv in self.dilated_convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.conv1x1(res)
        return res


class MiniDL(nn.Module):
    def __init__(self, in_ch):
        super(MiniDL, self).__init__()
        self.aspp = ASPP(in_ch, dilations=[1, 6])

    def forward(self, x):
        return self.aspp(x)


if __name__ == '__main__':
    a = torch.randn(1, 1, 64, 64, 64)
    aspp = ASPP(1)
    print(aspp(a).size())
    # gap = GAP(1)
    # print(gap(a).size())