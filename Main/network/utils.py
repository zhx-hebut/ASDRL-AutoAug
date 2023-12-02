import time
import torch
from torch.nn import functional as F


def get_prediction_from_logits(logits):
    size = logits.size()
    if size[1] > 1:
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    else:
        size = list(size)
        # delete channel dim to prevent
        size = [size[0]] + size[2:]
        preds = torch.round(torch.sigmoid(logits)).long().reshape(size)
    return preds


def make_same_size(logits, target):
    assert isinstance(logits, torch.Tensor), "model output {}".format(type(logits))
    size = logits.size() 
    if logits.size() != target.size():
        if len(size) == 5:
            logits = F.interpolate(logits, target.size()[2:], align_corners=False, mode='trilinear')
        elif len(size) == 4:
            logits = F.interpolate(logits, target.size()[2:], align_corners=False, mode='bilinear')
        else:
            raise Exception("Invalid size of logits : {}".format(size))
    return logits


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() <= nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass


def get_prediction(logits, should_sigmoid=True):
    size = logits.size()
    if should_sigmoid:
        # N x 1 x H x W
        if size[1] > 1:
            pred = F.softmax(logits, dim=1) 
            nclass = size[1]
        else:
            pred = F.sigmoid(logits)
            pred = torch.cat([1 - pred, pred], 1) #*-*
            nclass = 2
    else:
        pred = logits
        nclass = size[1]
    return pred, nclass


def get_numpy(tensor_list):
    res = []
    for tensor in tensor_list:
        if isinstance(tensor, torch.Tensor):
            res.append(tensor.detach().cpu().numpy())
        else:
            res.append(tensor)
    return res


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def get_slice(imgs, d):
    res = []
    for img in imgs:
        res.append(img[d])
    return res