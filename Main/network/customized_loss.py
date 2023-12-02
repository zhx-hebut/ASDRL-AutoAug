import numpy as np
import torch
import torch.nn as nn
from network.loss_funs import *


"""
For log sake...
Note : loss can only be scala type.
"""


def tensor_to_scala(data):
    if isinstance(data, list) or isinstance(data, tuple):
        new_data = []
        for l in data:
            if isinstance(l, torch.Tensor):
                new_data.append(l.item())
            else:
                new_data.append(l)
    elif isinstance(data, torch.Tensor):
        new_data = data.item()
    else:
        new_data = data
    return new_data


class LoggedLoss(nn.Module):
    def __init__(self, names=[], max_length=500):
        super(LoggedLoss, self).__init__()
        self.cached_total_loss = []
        self.cached_split_loss = []
        self.names = names
        self.max_length = max_length

    def forward(self, logits, target):
        raise NotImplementedError()

    def append_loss(self, total_loss, split_loss):
        """ total_loss and split_loss should be float or int , not tensor
        :param total_loss:
        :param split_loss:
        :return:
        """
        total_loss = tensor_to_scala(total_loss)
        split_loss = tensor_to_scala(split_loss)

        # assert not isinstance(total_loss, torch.Tensor), "total_loss and split_loss should be float or int , not tensor"

        if isinstance(split_loss, list) or isinstance(split_loss, tuple):
            self.cached_split_loss.append(split_loss)
        else:
            self.cached_split_loss.append([split_loss])

        self.cached_total_loss.append(total_loss)
        if len(self.cached_total_loss) == self.max_length:
            self.cached_total_loss.pop(0)

    def get_split_loss(self):
        return list(np.mean(self.cached_split_loss, 0))

    def get_loss(self):
        return np.mean(self.cached_total_loss)

    def get_last(self):
        return self.cached_total_loss[-1]

    def get_last_split(self):
        return list(self.cached_split_loss[-1])

    def get_loss_length(self):
        return len(self.names)

    def clear_cache(self):
        self.cached_total_loss = []
        self.cached_split_loss = []

    def get_loss_names(self):
        return self.names


class SingleLossWrapper(LoggedLoss):
    """ A wrapper for single loss function to record loss.

        To help user know how many loss get for an epoch,
        this wrapper is created to hold the origin loss function
        and record its output loss simultaneously.
    """
    def __init__(self, loss_func):
        super(SingleLossWrapper, self).__init__()
        self.loss_func = loss_func

    def forward(self, logits, target):
        loss = self.loss_func(logits, target)
        # l = loss
        # if isinstance(loss, torch.Tensor):
        #     l = loss.item()
        # self.append_loss(l, l)
        return loss


class MultiLossWrapper(LoggedLoss):
    """ A wrapper for multi loss function to record multi losses.

      Sometimes, a network will return many outputs, then use different
      loss functions to compute final loss. Thus, this class putes
      these loss functions together to generate a single output.

      Also it records loss history to print to console.
    :param
      loss_funcs : a loss function list, [func1, func2...]
      weights    : a list of weights assigned for corresponding loss
      names      : loss function names, default is [loss_1, loss_2, ...]
    """
    def __init__(self, loss_funcs, weights=None, names=[]):
        if names == []:
            for i in range(len(loss_funcs)):
                names.append('loss_' + str(i))
        super(MultiLossWrapper, self).__init__(names=names)
        if weights is None:
            self.weights = [1.] * len(loss_funcs)
        else:
            assert len(weights) == len(loss_funcs)
            self.weights = weights

        self.loss_funcs = nn.ModuleList(loss_funcs)

    def forward(self, inputs, target):
        assert len(inputs) == len(self.loss_funcs)

        total_loss = 0
        losses = []
        for i in range(len(self.loss_funcs)):
            loss_func = self.loss_funcs[i]
            weight = self.weights[i]
            loss = loss_func(inputs[i], target) * weight
            total_loss += loss
            losses.append(loss.item())
        self.append_loss(total_loss, losses)
        return total_loss


def ce_loss(nclass):
    return SingleLossWrapper(WeightedCE(nclass))


def bce_loss(nclass):
    return SingleLossWrapper(BCELoss(nclass))


def dice_loss(nclass):
    return SingleLossWrapper(DiceLoss(nclass))

def bcedice_loss():
    return SingleLossWrapper(BCEDiceLoss())

def simplified_dice(nclass):
    return SingleLossWrapper(SimplifiedDice(nclass))


def focal_loss(nclass):
    return SingleLossWrapper(FocalLoss(nclass, gamma=1))


def dice_bound_loss(nclass):
    return SingleLossWrapper(BoundedDice(nclass))


def mse_loss(nclass):
    import torch
    assert nclass == 1, 'not support multi class'
    return torch.nn.MSELoss()

