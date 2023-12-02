
import cv2
import random
import numpy as np
from imgaug import augmenters as iaa
from torchvision.transforms import functional as F
import torch

__all__ = ['resize', 'to_tensor', 'flip', 'rotation', 'crop', 'zoom', 'warp',
           'add_GaussianNoise', 'equalize_hist', 'sharpen', 'smooth', 'color']

def resize(img, mask, shape=(224, 224)):
    img = cv2.resize(img, shape)
    mask = cv2.resize(mask, shape)
    return img, mask

def to_tensor(img, mask, img_type=torch.float32, mask_type=torch.long):
    img = F.to_tensor(img).type(img_type)
    mask = np.expand_dims(mask, 0)
    mask = torch.from_numpy(mask).type(mask_type)
    return img, mask
    pass

def flip(img, mask, direction='h'):
    if direction == 'h' or direction == 'both':
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    if direction == 'v' or direction == 'both':
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)
    return img, mask

def rotation(img, mask, angle=15):
    rows, cols = img.shape[:2]
    m = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, m, (cols, rows))
    mask = cv2.warpAffine(mask, m, (cols, rows))
    return img, mask

def crop(img, mask, direction='x', step=20):
    if direction == 'x':
        aug = iaa.Affine(translate_px={"x": step})
    elif direction == 'y':
        aug = iaa.Affine(translate_px={"y": step})
    elif direction == 'both':
        aug = iaa.Affine(translate_px=step)
    img = aug.augment_image(img)
    mask = aug.augment_image(mask)
    return img, mask


def zoom(img, mask, rate=1.1):
    aug = iaa.Affine(scale=rate)
    img = aug.augment_image(img)
    mask = aug.augment_image(mask)
    return img, mask

def warp(img, mask):
    # scale = (0.01, 0.02)
    # aug = iaa.PiecewiseAffine(scale=(0.01, 0.02))
    value = random.random() / 50
    aug = iaa.PiecewiseAffine(value)
    img = aug.augment_image(img)
    mask = aug.augment_image(mask)
    return img, mask

def add_GaussianNoise(img, mask, rate=0.05):
    aug = iaa.AdditiveGaussianNoise(scale=rate * 255)
    if np.max(img) > 1:
        img = aug.augment_image(img)
    else:
        img = img * 255
        img = img.astype(np.uint8)
        img = aug.augment_image(img)
        img = img / 255
    return img, mask

def equalize_hist(img, mask):
    img = img * 255
    img = img.astype(np.uint8)
    if img.ndim == 3:
        for i in range(img.shape[-1]):
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])
    else:
        img = cv2.equalizeHist(img)
    return img/255, mask


def sharpen(image, mask):
    image = image * 255
    image = image.astype(np.uint8)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst/255, mask

def smooth(image, mask):
    kernel = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], np.float32)*0.1  
    dst = cv2.filter2D(image, -1, kernel=kernel)
    # cv2.imshow("custom_blur_demo", dst)
    return dst, mask

def color(img, mask, a=1., b=0.):
    if np.max(img) > 1:
        img = img * a + b
        img = np.clip(img, 0, 255)
    else:
        img = img * a + b
        img = np.clip(img, 0, 1.)
    return img, mask

