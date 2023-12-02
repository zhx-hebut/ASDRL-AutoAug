# -*- coding:utf-8 -*-
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR


__all__ = ['PolyLR', 'StepLR', 'CosineLR']


class WarmUpLRScheduler(object):
    def __init__(self, optimizer, total_epoch, iteration_per_epoch, warmup_epochs=0, iteration_decay=True):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.total_epoch = total_epoch
        self.iteration_per_epoch = iteration_per_epoch
        self.total_iteration = total_epoch * iteration_per_epoch
        self.warmup_epochs = warmup_epochs
        self.iteration_decay = iteration_decay

        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))

        self.step(0, 0)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self, epoch, iter):
        raise NotImplementedError

    def step(self, epoch, iter):
        # decay with epoch
        if not self.iteration_decay:
            iter = 0

        # get normal iteration
        lr_list = self.get_lr(epoch, iter) 

        # current iteration
        T = epoch * self.iteration_per_epoch + iter
        # warm up
        if self.warmup_epochs > 0 and T > 0 and epoch < self.warmup_epochs:
            # start from first iteration not 0
            lr_list = [lr * 1.0 * T /
                       (self.warmup_epochs*self.iteration_per_epoch) for lr in self.base_lrs]

        # adjust learning rate for all groups
        for param_group, lr in zip(self.optimizer.param_groups, lr_list):
            param_group['lr'] = lr
        return lr

class PolyLR(WarmUpLRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    multiply by (1 - iter / total_iter) ** gamma.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        power (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        total_iter(int) : Total epoch
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, total_epoch, iteration_per_epoch, warmup_epochs=0, iteration_decay=True,
                 power=0.9):
        self.power = power
        super(PolyLR, self).__init__(optimizer, total_epoch, iteration_per_epoch, warmup_epochs, iteration_decay)

    def get_lr(self, epoch, iter):
        T = epoch * self.iteration_per_epoch + iter
        return [base_lr * ((1 - 1.0 * T / self.total_iteration) ** self.power)
                for base_lr in self.base_lrs]


class StepLR(WarmUpLRScheduler):
    """

    Step decay : ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Example : step = 30
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, total_epoch, iteration_per_epoch, warmup_epochs=0, iteration_decay=True,
                 step_size=2, gamma=0.1):
        self.lr_step = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, total_epoch, iteration_per_epoch, warmup_epochs, iteration_decay)

    def get_lr(self, epoch, iter):
        return [base_lr * (self.gamma ** (epoch // self.lr_step))
                for base_lr in self.base_lrs]


class CosineLR(WarmUpLRScheduler):
    """
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter * pi))``
    """
    def __init__(self, optimizer, total_epoch, iteration_per_epoch, warmup_epochs=0, iteration_decay=True,):
        super(CosineLR, self).__init__(optimizer, total_epoch, iteration_per_epoch, warmup_epochs, iteration_decay)

    def get_lr(self, epoch, iter):
        T = epoch * self.iteration_per_epoch + iter
        return [0.5 * base_lr * (1 + math.cos(1.0 * T / self.total_iteration * math.pi))
                for base_lr in self.base_lrs]


class NoneLR(WarmUpLRScheduler):
    """
    Do nothing...
    """
    def __init__(self, optimizer, total_epoch, iteration_per_epoch, warmup_epochs=0, iteration_decay=True,):
        super(NoneLR, self).__init__(optimizer, total_epoch, iteration_per_epoch, warmup_epochs, iteration_decay)

    def get_lr(self, epoch, iter):
        return self.base_lrs
