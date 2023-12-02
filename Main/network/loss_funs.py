import torch
import torch.nn as nn
import numpy as np
from network.utils import make_same_size, to_one_hot, get_prediction
import torch.nn.functional as F
__all__ = ['WeightedCE', 'WeightedMSE', 'BCELoss', 'DiceLoss','BCEDiceLoss', 'BoundedDice', 'GeneralizedDiceLoss', 'FocalLoss', 'SimplifiedDice']


class WeightedCE(nn.Module):
    def __init__(self, nclass, weights=[1, 1]):
        super(WeightedCE, self).__init__()
        self.bc = nn.CrossEntropyLoss(weight=torch.Tensor(weights))

    def forward(self, logits, target):
        target = target[:, 0, ...]
        return self.bc(logits, target)


class BCELoss(nn.Module):
    def __init__(self, nclass):
        super(BCELoss, self).__init__()
        self._bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        # logits = make_same_size(logits, target)
        logits_flatten = logits.view(-1).type(torch.float32)
        labels_flatten = target.view(-1).type(torch.float32)
        return self._bce_loss(logits_flatten, labels_flatten)


class DiceLoss(nn.Module): # *.*
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth 
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def forward(self, logits, target):
        size = logits.size()
        N, nclass = size[0], size[1]
        pre = torch.sigmoid(logits)
        num = target.size(0)
        pre = pre.view(num, -1)
        target = target.view(num, -1)
        intersection = (pre * target)
        dice = (2. * intersection.sum(1) + self.smooth) / (pre.sum(1) + target.sum(1) + self.smooth)
        loss = 1 - dice.sum() / num
        return  loss

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target.float())#
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class BoundedDice(nn.Module):
    def __init__(self, nclass):
        super(BoundedDice, self).__init__()
        self.dice = DiceLoss(nclass)

    def forward(self, logits_list, target_list):
        total_loss = 0
        num_loss = 0
        for i in range(len(logits_list)):
            logits = logits_list[i]
            target, bound = target_list[1][i]
            (min_x, min_y, min_z, max_x, max_y, max_z) = bound
            if min_x == max_x:
                continue
            target = target[None, :, min_z:max_z, min_y:max_y, min_x, max_x]
            loss = self.dice(logits, target)
            total_loss += loss
            num_loss += 1
        return total_loss / num_loss


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, nclass):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, logits, target):
        logits = make_same_size(logits, target)

        size = logits.size()
        N, nclass = size[0], size[1]

        pred, nclass = get_prediction(logits, should_sigmoid=True)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot**2 + target_one_hot**2

        # N x C
        inter = inter.view(N, nclass, -1).sum(2).sum(0)
        union = union.view(N, nclass, -1).sum(2).sum(0)

        # NxC
        dice = 2 * inter / union
        return dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, nclass, gamma=2.):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        logits = make_same_size(logits, target)

        N = logits.size()[0]
        pred, nclass = get_prediction(logits, should_sigmoid=True)

        # N x C x H x W
        prob = pred
        target = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        target = target.float()
        prob = prob.view(N, -1).clamp(1e-10, 0.999999)# prevent nan -inf
        target = target.view(N, -1)
        loss = (target * ((1-prob)**self.gamma) * prob.log()) + \
               ((1-target)*((prob**self.gamma)*((1-prob).log())))
        return -loss.mean()


class SimplifiedDice(nn.Module):
    def __init__(self, nclass, smooth=1e-5):
        super(SimplifiedDice, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        logits = make_same_size(logits, target)
        target = target.float()
        size = logits.size()
        assert size[1] == 1 # only support binary case
        logits = logits.view(size[0], -1)
        target = target.view(size[0], -1)
        inter = (logits * target).sum(1)
        union = (logits + target).sum(1)
        dice = (2*inter+self.smooth) / (union+self.smooth)
        return -dice.mean()


class WeightedMSE(nn.Module):
    def __init__(self, nclass):
        super(WeightedMSE, self).__init__()

    def forward(self, logits, target):
        pos_loss = ((logits[target>0] - target[target>0]) **2) / (target>0).sum()
        neg_loss = ((logits[target<=0] - target[target<=0]) **2) / (target<=0).sum()
        return (pos_loss.sum() + neg_loss.sum())/2 * logits.size()[0]

