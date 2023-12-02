import numpy as np
import torch
from torch.nn import functional as F
from network.utils import to_one_hot, make_same_size, get_probability
import scipy.spatial.distance as dist
import skimage.segmentation as skseg
from scipy.ndimage import measurements
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 


__all__ = ['IOU', 'DICE', 'mean_iou', 'mean_dice', 'SEN', 'PPV']


def get_confusion_matrix(prediction:np.ndarray, target:np.ndarray, nclass):
    """ Compute confusion matrix for prediction size=(nclass, nclass)
    :param prediction: (N, H, W) ndarray
    :param target:     (N, H, W) ndarray
    :param nclass:
    :return:
    """
    assert prediction.shape == target.shape, \
        "Shape mismatch pred.shape={}, target.shape={}".format(prediction.shape, target.shape)
    mask = (target >= 0) & (target < nclass)
    label = nclass * target[mask].astype('int') + prediction[mask]
    count = np.bincount(label, minlength=nclass ** 2)
    confusion_matrix = count.reshape(nclass, nclass)
    return confusion_matrix


def pixel_acc(confusion_matrix:np.ndarray):
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()


def pixel_acc_per_class(confusion_matrix:np.ndarray):
    acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    return np.nanmean(acc)


def mIoU(confusion_matrix:np.ndarray):
    inter = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - inter
    iou = inter / union
    return np.nanmean(iou)


def FWIoU(confusion_matrix:np.ndarray):
    """ Frequency weighted mean IoU
    :param confusion_matrix:
    :return:
    """
    iou = mIoU(confusion_matrix)
    freq = confusion_matrix.sum(axis=1) / confusion_matrix.sum()
    FWIoU = (freq[freq>0]*iou[freq>0]).sum()
    return FWIoU


def hausdorff_distance(prediction:np.ndarray, target:np.ndarray):
    """ Compute Hausdorff distance. Takes about 0.7min for 422*6 images
    :param prediction: shape=(N, H, W)
    :param target:     shape=(N, H, W)
    :return:
    """
    assert prediction.shape == target.shape, \
        "Shape mismatch pred.shape={}, target.shape={}".format(prediction.shape, target.shape)
    assert 0 <= prediction.max() <= 1
    hd_list = []
    for pred_slice, target_slice in zip(prediction, target):
        pred_coord = np.array(np.where(pred_slice == 1)).transpose(1, 0)
        target_coord = np.array(np.where(target_slice == 1)).transpose(1, 0)
        hd_list.append(dist.directed_hausdorff(pred_coord, target_coord)[0])
    return np.nanmean(hd_list)


def average_surface_distance(prediction, target):
    """ Compute Average surface distance. Takes about 1.2 min for 422*6 images (2.34->3.49)
    :param prediction: shape=(N, H, W)
    :param target:     shape=(N, H, W)
    :return:
    """
    assert prediction.shape == target.shape, \
        "Shape mismatch pred.shape={}, target.shape={}".format(prediction.shape, target.shape)
    asd_list = []
    for pred_slice, target_slice in zip(prediction, target):
        # have to find boundaries to reduce for loop computation cost
        pred_slice = skseg.find_boundaries(pred_slice, connectivity=1, mode='thick', background=0).astype(np.float32)
        target_slice = skseg.find_boundaries(target_slice, connectivity=1, mode='thick', background=0).astype(np.float32)

        pred_coord = np.array(np.where(pred_slice == 1)).transpose(1, 0)
        target_coord = np.array(np.where(target_slice == 1)).transpose(1, 0)
        if len(pred_coord) == 0 or len(target_coord) == 0:
            asd_list.append(0)
        else:
            asd_list.append(compute_asd(pred_coord, target_coord))
    return np.nanmean(asd_list)


def compute_asd(pred_coord, target_coord):
    min_distance_sum = 0
    for p in pred_coord:
        distance = ((p - target_coord)**2)
        min_distance_sum += distance.min()

    for t in target_coord:
        distance = ((t - target_coord)**2)
        min_distance_sum += distance.min()

    N1 = pred_coord.shape[0]
    N2 = target_coord.shape[0]
    if N1 + N2 == 0:
        return 0
    else:
        return min_distance_sum / (N1+N2)


def centroid_distance(prediction, target):
    """ Compute Average surface distance. Takes about 1.2 min for 422*6 images (2.34->3.49)
    :param prediction: shape=(N, H, W)
    :param target:     shape=(N, H, W)
    :return:
    """
    assert prediction.shape == target.shape, \
        "Shape mismatch pred.shape={}, target.shape={}".format(prediction.shape, target.shape)
    cd_list = []
    for pred_slice, target_slice in zip(prediction, target):
        pred_com = np.array(measurements.center_of_mass(pred_slice))
        target_com = np.array(measurements.center_of_mass(target_slice))
        d = ((pred_com - target_com)**2).sum()
        cd_list.append(d)
    return np.nanmean(cd_list)


def get_inter_union(logits, target, should_sigmoid=True):
    """
    :param logits: N x C x H x W
    :param target: N x 1 x H x W
    :return:
    """
    # if torch.is_tensor(logits):
    #     logits = torch.sigmoid(logits).data.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target.data.cpu().numpy()

    logits = torch.sigmoid(logits)
    logits = logits > 0.5
    target = target > 0.5
    num = target.shape[0]
    logits = logits.reshape(num,-1)
    target = target.reshape(num,-1)
    inter = (logits * target).sum(1)
    union = logits.sum(1) + target.sum(1) - inter
    return inter, union


# iou = (class_inter + self.eps) / (class_union + self.eps)
def IOU(inter, union):
    # size = [class,]
    eps = 1e-5
    res = (inter + eps) / (union + eps)
    # mark : background is not included
    return res


# dice = (2 * class_inter + self.eps) / (class_union + class_inter)
def DICE(inter, union):
    # size = [class,]
    eps = 1e-5
    res = (2 * inter + eps) / (union + inter + eps)
    # mark : background is not included
    return res


def mean_iou(logits, target):
    """
    mean IOU
    :param logits: NxCxHxW
    :param target: Nx1xHxW
    :return:
    """
    # N x C
    inter, union = get_inter_union(logits, target)
    return IOU(inter, union)


def mean_dice(logits, target):
    """
    mean dice
    :param logits: N x C x H x W
    :param target: N x 1 x H x W
    :return:
    """
    # N x C
    inter, union = get_inter_union(logits, target)
    return DICE(inter, union)


def cosine_similarity(logits, target):
    target = target.float()
    logits = make_same_size(logits, target)
    sim = (logits * target).sum() / ((logits ** 2).sum().sqrt() * (target ** 2).sum().sqrt())
    return (sim + 1) / 2


def mean_square_similarity(logits, target):
    logits = make_same_size(logits, target)
    target = target.float()
    return 1 - ((logits - target) ** 2).mean()


def SEN(logits, target):
    logits = make_same_size(logits, target)
    eps = np.spacing(1)
    # NxC
    inter, union = get_inter_union(logits, target)
    # NxC
    N, C, W, H = target.size()
    target = target.view(N, C, -1).sum(2)
    # NxC
    sen = ((inter + eps) / (target + eps))
    # remove background
    return sen[:, 1:].mean()


def PPV(logits, target):
    logits = make_same_size(logits, target)
    eps = np.spacing(1)
    # NxC
    pred, _ = get_probability(logits)
    # NxC
    inter, union = get_inter_union(logits, target)

    N, C, W, H = pred.size()
    pred = pred.view(N, C, -1).sum(2)

    # NxC
    ppv = ((inter + eps) / (pred + eps))
    # remove background
    return ppv[:, 1:].mean()

