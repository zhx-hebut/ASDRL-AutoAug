import torch
from torch import nn as nn

import numpy as np

from network.evaluate_funcs import IOU, DICE, get_inter_union, mean_iou, mean_dice, get_confusion_matrix, \
    hausdorff_distance, average_surface_distance, centroid_distance
from network.utils import make_same_size, get_prediction

__all__ = ['LoggedMeasure', 'SegMeasure', 'MultiSegMeasure', 'MedicalImageMeasure']

"""
Note that Measure cache can be any type.
"""


class LoggedMeasure(nn.Module):
    """ Like loggedLoss as a wrapper of loss function, is is also a
    wrapper for evaluation function to log evaluation.
    See **SegMeasure** for detail implementation.
    :param names : names for your outputs,
    :param main_acc_name : main measurement of evaluation ( of model performance). default : first name
    :param max_length : max history of evaluation
    """
    def __init__(self, names, main_acc_name=None, max_length=500):
        super(LoggedMeasure, self).__init__()
        self.names = names
        self.main_acc_axis = 0 if main_acc_name is None else self.names.index(main_acc_name.lower())
        self.max_length = max_length 
        self.caches = []

    def clear_cache(self):
        self.caches.clear()

    def append_eval(self, evaluation):
        """ Append evaluation to caches
        :param evaluation : the result of one evaluation
        """
        self.caches.append(evaluation)
        if self.training and len(self.caches) > self.max_length:
            self.caches.pop(0)

    def get_column(self, caches, idx):
        """ Get column from a 2-d(dim>=2) array
        :param caches:
        :param idx:
        :return:
        """
        column = []
        for cache in caches:
            column.append(cache[idx])
        return column

    def forward(self, logits, target):
        """ Calculate accuracy. Should be implemented by user.
        :param logits: output from network
        :param target: label from input
        """
        raise NotImplemented()

    def get_acc(self):
        """ Should return the accuracy(average acc is recommended).
        """
        raise NotImplemented()

    def get_last(self):
        """ Should return the last accuracy.
        """
        raise NotImplemented()

    def get_acc_length(self):
        return len(self.names)

    def get_main_acc_idx(self):
        return self.main_acc_axis

    def get_acc_names(self):
        return self.names

    def get_main_acc(self):
        return self.get_acc()[self.main_acc_axis]

    def set_max_len(self, max_len):
        self.max_length = max_len

    def get_max_len(self):
        return self.max_length


class EmptyMeasure(LoggedMeasure):
    def __init__(self):
        super(EmptyMeasure, self).__init__(names=['empty'])

    def forward(self, logits, target):
        return 0.,

    def get_acc(self):
        return 0.,

    def get_last(self):
        return 0.,


class SegMeasure(LoggedMeasure):
    """The channel of input logits should be the same as class number
    :param reduction  : the type of calculate accuracy of the result, {'immediate', 'delayed'},
                        'immediate' means average calculated acc,
                        'delayed'   means save the intersection and union, then calculate them
    :param main_acc_name : specify which accuracy to use as final evaluation, {'iou', 'dice'}
    :param multi_inputs_axis : if there is multiple outputs from the network,
                               use which output to compute result
    """

    def __init__(self, reduction='delayed', main_acc_name='iou',
                 multi_inputs_axis=-1, max_length=500):
        names = ['iou', 'dice']
        super(SegMeasure, self).__init__(names, main_acc_name=main_acc_name, max_length=max_length)
        self.iou = mean_iou
        self.dice = mean_dice
        self.reduction = reduction 
        self.multi_inputs_axis = multi_inputs_axis 
        self.clear_cache() 

    def forward(self, logits, target):
        if self.multi_inputs_axis != -1:
            # logits is a tuple object
            logits = logits[self.multi_inputs_axis]

        if isinstance(logits, list) or isinstance(logits, tuple):
            raise Exception("multi_inputs_axis must specified for multi outputs")

        logits = make_same_size(logits, target)
        if self.reduction == 'immediate':
            iou = self.iou(logits, target)
            dice = self.dice(logits, target)
            # calculate then average
            self.append_eval((iou.view(1), dice.view(1)))
        else:
            inter, union = get_inter_union(logits, target)
            # save then calculate
            self.append_eval((inter, union))
        # pre = torch.sigmoid(logits)
        # num = target.size(0)
        # pre = pre.view(num, -1)
        # target = target.view(num, -1)
        # intersection = (pre * target)
        # dice = (2. * intersection.sum(1) + self.smooth) / (pre.sum(1) + target.sum(1) + self.smooth)

    def get_acc(self):
        col_0 = self.get_column(self.caches, 0)
        col_1 = self.get_column(self.caches, 1)
        if self.reduction == 'immediate':
            ious, dices = col_0, col_1
            iou = torch.cat(ious, 0)
            dice= torch.cat(dices, 0)
        else:
            # size : C
            inter, union = col_0, col_1
            class_inter = torch.cat(inter, 0)
            class_union = torch.cat(union, 0)
            iou = IOU(class_inter, class_union)
            dice = DICE(class_inter, class_union)
        return iou.mean().item(), dice.mean().item()

    def get_last(self):
        # last_cache = self.caches[-1]
        # if self.reduction == 'immediate':
        #     iou, dice = last_cache
        # else:
        #     inter, union = last_cache
        #     iou = IOU(inter, union)
        #     dice = DICE(inter, union)
        iou = []
        dice = []
        for i in self.caches:
            num = len(i[0])
            a = i[0]
            b = i[1]
            for j in range(num):
                inter, union = a[j],b[j]
                iou.append(IOU(inter, union).item())
                dice.append(DICE(inter, union).item())
        return sum(iou)/len(iou), sum(dice)/len(dice)


class SimpleMeasure(LoggedMeasure):
    def __init__(self, name, func, max_length=500):
        super(SimpleMeasure, self).__init__([name], max_length=max_length)
        self.func = func

    def forward(self, logits, target):
        acc = self.func(logits, target)
        self.append_eval(acc)

    def get_acc(self):
        return torch.stack(self.caches, 0).mean(),

    def get_last(self):
        return self.caches[-1],


class DiceMeasure(LoggedMeasure):

    def __init__(self, name='iou', reduction='delayed', max_length=500):
        name = name.lower()
        if name == 'iou':
            names = ['iou']
            self.func = mean_iou
            self.delay_func = IOU
        elif name == 'dice':
            names = ['dice']
            self.func = mean_dice
            self.delay_func = DICE
        else:
            raise Exception("Error name : {}".format(name))

        super(DiceMeasure, self).__init__(names, max_length=max_length)
        self.reduction = reduction
        self.clear_cache()

    def forward(self, logits, target):
        """
        :type  logits: torch.Tensor : size is [N, C, D, H, W]
        :type target: torch.Tensor : size is [N, 1, D, H, W]
        """
        logits = make_same_size(logits, target)
        if self.reduction == 'immediate':
            acc = self.func(logits, target)
            # calculate then average
            self.append_eval((acc.view(1),))
        else:
            inter, union = get_inter_union(logits, target)
            # save then calculate
            self.append_eval((inter, union))

    def get_acc(self):
        if self.reduction == 'immediate':
            acc_list = self.get_column(self.caches, 0)
            acc = torch.cat(acc_list, 0)
        else:
            col_0 = self.get_column(self.caches, 0)
            col_1 = self.get_column(self.caches, 1)
            # size : C
            inter, union = col_0, col_1
            class_inter = torch.cat(inter, 0)
            class_union = torch.cat(union, 0)
            acc = self.delay_func(class_inter, class_union)
        return acc.mean().item(),

    def get_last(self):
        last_cache = self.caches[-1]
        if self.reduction == 'immediate':
            acc = last_cache
        else:
            inter, union = last_cache
            acc = self.delay_func(inter, union)
        return acc.mean().item(),


class MultiSegMeasure(LoggedMeasure):
    """
    :type list[LoggedMeasure] measures : a list[LoggedMeasure]
    """
    def __init__(self, measures, main_acc_name=None, max_length=500):
        self.measures = measures
        self.names = []
        for measure in self.measures:
            self.names += measure.names
        super(MultiSegMeasure, self).__init__(self.names, main_acc_name=main_acc_name, max_length=max_length)

    def forward(self, logits, target):
        assert len(logits) == len(target) == len(self.measures)

        for measure, l, t in zip(self.measures, logits, target):
            measure(l, t)

    def get_acc(self):
        acc_list = []
        for measure in self.measures:
            for acc in measure.get_acc():
                acc_list.append(acc)
        return acc_list

    def get_last(self):
        acc_list = []
        for measure in self.measures:
            for acc in measure.get_last():
                acc_list.append(acc)
        return acc_list

    def clear_cache(self):
        for measure in self.measures:
            measure.clear_cache()


class Class3SegMeasure(SegMeasure):
    def __init__(self, reduction='delayed', main_acc_name='iou', max_length=500):
        super(Class3SegMeasure, self).__init__(main_acc_name=main_acc_name, reduction=reduction, max_length=max_length)

    def forward(self, logits, target):
        if target.max() < 2:
            new_logits = logits[:, [0, 2], ...]
            new_logits[:, 0, ...] += logits[:, 1, ...]
        else:
            new_logits = logits
        super(Class3SegMeasure, self).forward(new_logits, target)


class MedicalImageMeasure(LoggedMeasure):
    """
    mIoU is mean Intersection of Union
    DSC  is dice coefficiency
    ACC  is total accuracy
    HD   is hausdorff distance
    PPV  is positive predictive value
    SEN  is sensitivity or true positive rate
    ASD  is average surface distance
    CD   is centroid distance
    """
    def __init__(self, nclass, max_length=500):
        super(MedicalImageMeasure, self).__init__(nclass, names=['DSC', 'mIoU', 'ACC', 'PPV', 'SEN', 'CD', 'ASD', 'HD'],
                                                  main_acc_name='DSC', max_length=max_length)
        self.confusion_matrix = np.zeros((nclass, nclass))
        self.current_matrix = None

    def clear_cache(self):
        self.confusion_matrix = np.zeros((self.nclass, self.nclass))
        self.caches = []

    def forward(self, logits, target):
        prediction = get_prediction(logits)
        if len(target.size()) != len(prediction.size()):
            target = target[:, 0]
        prediction = prediction.cpu().numpy()
        target     = target.cpu().numpy()

        self.current_matrix = get_confusion_matrix(prediction, target, self.nclass)
        self.confusion_matrix += self.current_matrix
        cd = centroid_distance(prediction, target)
        asd = average_surface_distance(prediction, target)
        hd = hausdorff_distance(prediction.copy(), target.copy())
        self.append_eval((cd, asd, hd))

    def get_res(self, matrix):
        # the confusion matrix is sort by class, so first 0(neg), then 1(pos)
        TN, FN, FP, TP = matrix.ravel()

        DCS = (2*TP) / (FP+TP*2+FN)
        IoU = TP / (TP+FN+FP)
        ACC = (TP+TN) / matrix.sum()
        PPV = TP / (TP+FP)
        SEN = TP / (TP+FN)
        CD = np.nanmean(self.get_column(self.caches, 0))
        ASD = np.nanmean(self.get_column(self.caches, 1))
        HD = np.nanmean(self.get_column(self.caches, 2))
        return DCS, IoU, ACC, PPV, SEN, CD, ASD, HD

    def get_acc(self):
        return self.get_res(self.confusion_matrix)

    def get_last(self):
        return self.get_res(self.current_matrix)


if __name__ == '__main__':
    pred = np.array([[1, 1, 0],
                      [1, 1, 0],
                      [0, 0, 0]])

    ground = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [0, 1, 1]])
    matrix = get_confusion_matrix(pred, ground, 2)
    print(matrix)
