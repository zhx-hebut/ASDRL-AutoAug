import torch
from PIL import Image, ImageFilter
from PIL.Image import BILINEAR, NEAREST
from torchvision.transforms import functional as F
import numpy.random as random
import numpy as np

# from preprocess.visualize import show_graphs *-*

__all__ = ['compose', 'resize', 'random_crop', 'random_flip',
           'random_gaussian_blur', 'random_rotate', 'random_scale',
           'to_tensor', 'normalize']


def compose(functions):
    """ Compose functions below
    :param functions:
    :return:
    Usage :
    >>> compose([resize([1, 2], BILINEAR), random_flip()])
    """

    def do_transform(img, mask):
        for func in functions: #to_tensor
            img, mask = func(img, mask)
        return img, mask

    return do_transform


def resize(size, mask_size=None):
    """Resize PIL Image size:(H, W)
    :param size:
    :param interpolation:
    :return:
    """

    def resize_img(img, mask):
        img = F.resize(img, size, BILINEAR)
        if mask_size is not None:
            mask = F.resize(mask, mask_size, NEAREST)
        return img, mask

    return resize_img


def random_flip(rand=0.5, direction='h', show=False):
    """Random flip PIL Image
    :param direction: ['h', 'v', 'both']
    :param show:
    :param rand:
    :return:
    """

    def random_flip_img(img, mask):
        if (direction == 'h' or direction == 'both') and random.random() < rand:
            img = F.hflip(img)
            mask = F.hflip(mask)
        if (direction == 'v' or direction == 'both') and random.random() < rand:
            img = F.vflip(img)
            mask = F.vflip(mask)
        # if show:
            # show_graphs([np.array(img), np.array(mask)])
        return img, mask

    return random_flip_img


def random_scale(scale_range=(0.75, 1.20), show=False):
    """Random scale PIL Image
    :param scale_range:
    :return:
    """

    def random_scale_img(img, mask):
        scale_factor = random.uniform(*scale_range)
        w, h = img.size
        if w > h:
            ratio = (1.0 * w) / h
            out_h = int(h * scale_factor)
            out_w = int(ratio * out_h)
        else:
            ratio = (1.0 * h) / w
            out_w = int(w * scale_factor)
            out_h = int(ratio * out_w)
        img = F.resize(img, (out_h, out_w), interpolation=BILINEAR)
        mask = F.resize(mask, (out_h, out_w), interpolation=NEAREST)
        # if show:
            # show_graphs([np.array(img), np.array(mask)])
        return img, mask

    return random_scale_img


def random_rotate(angle_range=(-10, 10), show=False):
    """Random rotate PIL Image
    :param angle_range:
    :return:
    """

    def random_rotate_img(img, mask):
        rotate_angle = random.uniform(*angle_range)
        img = F.rotate(img, rotate_angle, resample=BILINEAR)
        mask = F.rotate(mask, rotate_angle, resample=NEAREST)
        # if show:
            # show_graphs([np.array(img), np.array(mask)])
        return img, mask

    return random_rotate_img


def random_crop(size=[256, 256], img_fill=0, mask_fill=255, show=False):
    """Random crop PIL Image, it will pad the image first if the size is not enough
    :param size:
    :return:
    """

    def random_crop_img(img, mask):
        w, h = img.size
        # Padding img before crop if the size of img is to small
        if w <= size[0] or h <= size[1]:
            pad_w = size[0] - w if w < size[0] else 0
            pad_h = size[1] - h if h < size[1] else 0
            # pad around
            left_pad_w = pad_w // 2
            left_pad_h = pad_h // 2
            right_pad_w = pad_w - left_pad_w
            right_pad_h = pad_h - left_pad_h
            # left, top, right, bottom
            img = F.pad(img, (left_pad_w, left_pad_h, right_pad_w, right_pad_h), fill=img_fill)
            mask = F.pad(mask, (left_pad_w, left_pad_h, right_pad_w, right_pad_h), fill=mask_fill)
        w, h = img.size

        if w == size[0] and h == size[1]:
            return img, mask

        if h == size[1]:
            i = 0
        else:
            i = random.randint(0, h - size[1])

        if w == size[0]:
            j = 0
        else:
            j = random.randint(0, w - size[0])
        img = F.crop(img, i, j, size[1], size[0])
        mask = F.crop(mask, i, j, size[1], size[0])
        # if show:
            # show_graphs([np.array(img), np.array(mask)])
        return img, mask

    return random_crop_img


def random_gaussian_blur(rand=0.5):
    """Random gaussian blur with the possibility of rand
    :param rand: the possibility to blur image
    :return:
    """

    def random_gaussian_blur_img(img, mask):
        if random.random() < rand:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img, mask

    return random_gaussian_blur_img


def to_tensor(img_type=torch.float32, mask_type=torch.long):
    """ Convert PIL Image to torch tensor
    :return:
    """

    def to_tensor_img(img, mask):
        img1 = F.to_tensor(img)#.type(img_type)
        mask1 = F.to_tensor(mask)#.type(mask_type)
        # mask1 = np.array(mask)
        # mask1 = np.expand_dims(mask1, 0)
        # mask1 = torch.from_numpy(mask1).type(mask_type)
        return img1, mask1

    return to_tensor_img


def normalize():
    """ Normalize **tensor** image!
    :return:
    """

    def normalize_img(img, mask):
        img = torch.tensor(np.array(img)).unsqueeze(0)
        mask = torch.tensor(np.array(mask)).unsqueeze(0)

        channels = img.size()[0]
        if channels == 1:
            img = (img - img.min()) / (img.max() - img.min())
            if torch.sum(mask) != 0:
                mask = (mask - mask.min()) / (mask.max() - mask.min())
        else:
            # channel wise normalize
            for channel in range(channels):
                img[channel, :] = (img[channel, :] - img[channel, :].min()) / \
                                  (img[channel, :].max() - img[channel, :].min())
        return img, mask

    return normalize_img


def random_flip_numpy():
    def random_flip_3d(img, mask):
        if np.random.rand() < 0.5:
            img  = img[::-1, ::-1, ...]
            mask = mask[::-1, ::-1, ...]
        return img, mask
    return random_flip_3d


def random_zoom():
    '''
    enlarge and shrink (image, mask)
    :return:
    '''
    def random_zoom_mask(img, mask):
        rate = random.uniform(0.8, 1.4)
        w, h = mask.size
        new_w, new_h = int(rate * w), int(rate * h)
        img = F.resize(img, (new_w, new_h))
        mask = F.resize(mask, (new_w, new_h))
        img = F.center_crop(img, (w, h))
        mask = F.center_crop(mask, (w, h))
        return img, mask
    return random_zoom_mask


def random_shift():
    '''
    random shift (image, mask)
    :return:
    '''
    def random_shfit_location(img, mask):
        w, h = img.size
        left, upper, right, lower = mask.getbbox()
        center_w = int(left + right) / 2
        center_h = int(upper + lower) / 2
        new_center_w = random.randint(int(left/2), int((w-right)/2)+right)
        new_center_h = random.randint(int(upper/2), int((h-lower)/2)+lower)
        shift_w = int((center_w - new_center_w) / 2)
        shift_h = int((center_h - new_center_h) / 2)
        img = img.transform(img.size, Image.AFFINE, (1, 0, shift_w, 0, 1, shift_h))
        mask = mask.transform(img.size, Image.AFFINE, (1, 0, shift_w, 0, 1, shift_h))
        return img, mask
    return random_shfit_location
