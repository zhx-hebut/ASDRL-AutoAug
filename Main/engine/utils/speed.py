import time
from engine.utils import get_net_size, count_ops
import torch
import numpy as np


class Timer(object):
    def __init__(self, verbose=False):
        self.start_time = time.time()
        self.verbose = verbose
        self.duration = 0

    def restart(self):
        self.duration = self.start_time = time.time()
        return self.duration

    def stop(self):
        return time.time() - self.start_time

    def get_last_duration(self):
        return self.duration

    def __enter__(self):
        self.restart()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = self.stop()
        if self.verbose:
            print('{:^.4f} s'.format(self.stop()))


def to_cuda(data, device):
    if device < 0:
        return data
    else:
        return data.cuda(device)


def network_train_analyze(net, data_generate_func, cuda=0, train_time=10, forward_verbose=False):
    t1 = Timer(verbose=True)
    t2 = Timer(forward_verbose)
    t3 = Timer(verbose=False)
    if cuda >= 0:
        torch.cuda.reset_max_memory_allocated(cuda)
    forward_times = []
    backward_times = []
    with t1:
        for i in range(train_time):
            a = to_cuda(data_generate_func(), cuda)
            with t3:
                b = net(a)
                if forward_verbose:
                    print('forward  : ', end='')
            forward_times.append(t3.get_last_duration())

            with t2:
                b.sum().backward()
                if forward_verbose:
                    print('backward : ', end='')
            backward_times.append(t2.get_last_duration())
        print('total train time   : ', end='')
    print("Total iteration    : {}".format(train_time))
    print('mean forward  time : {:^.4f} s'.format(np.mean(forward_times[1:])))
    print('mean backward time : {:^.4f} s'.format(np.mean(backward_times[1:])))
    if cuda >= 0:
        print("Max memory allocated : {:^.4f} M".format(torch.cuda.max_memory_allocated(cuda) / (1024.**2)))


def network_test_analyze(net, data_generate_func, cuda=0, test_time=50, forward_verbose=False):
    t1 = Timer(verbose=True)
    t2 = Timer(verbose=forward_verbose)
    t3 = Timer(verbose=False)
    if cuda >= 0:
        torch.cuda.reset_max_memory_allocated(cuda)
    forward_times = []
    data_times = []
    with t1:
        with torch.no_grad():
            for i in range(test_time):
                with t3:
                    a = to_cuda(data_generate_func(), cuda)
                data_times.append(t3.get_last_duration())

                with t2:
                    net(a)
                    if forward_verbose:
                        print('forward  : ', end='')
                forward_times.append(t2.get_last_duration())
        print('total test time    : ', end='')
    print("Total iteration    : {}".format(test_time))
    print('mean data     time : {:^.4f} s'.format(np.mean(data_times[1:])))
    print('mean forward  time : {:^.4f} s'.format(np.mean(forward_times[1:])))
    if cuda >= 0:
        print("Max memory allocated : {:^.4f} M".format(torch.cuda.max_memory_allocated(cuda) / (1024.**2)))


def analyze_network_performance(net, data_generate_func, cuda=0, train_time=10, test_time=20, forward_verbose=False):

    print('============ Analyzing network performance ==============')

    print(get_net_size(net))

    net = to_cuda(net, cuda)
    a = data_generate_func()
    a = to_cuda(a, cuda)
    print(count_ops(net, a))

    print('-------------------Train analyze----------------')
    network_train_analyze(net, data_generate_func, cuda, train_time, forward_verbose)

    print('-------------------Test  analyze----------------')
    network_test_analyze(net, data_generate_func, cuda, test_time, forward_verbose)

