import numpy as np
from PIL import Image
from skimage import io
import warnings
import torch
from torch.utils.data.dataset import Dataset
warnings.filterwarnings('ignore')
from preprocess.tools import read_csv
import torchvision.transforms as transforms

__all__ = ['KidneyBaseDataSet']

def default_loader(path):
    return io.imread(path)

def RGB_loader(path):
    return Image.open(path).convert('RGB')

def gray_loader(path):
    img = Image.open(path).convert('L')
    # CenterCrop = transforms.CenterCrop((368,368))
    # cropped_image = CenterCrop(img)
    # return cropped_image
    return img

class KidneyBaseDataSet(Dataset):
    def __init__(self, cfg, transform=None, mode='train', loader=gray_loader):
        super(KidneyBaseDataSet, self).__init__()
        self.mode = mode
        self.transform = transform
        self.loader = loader

        train_data = read_csv(cfg.datasets.train_data_dir)
        val_data = read_csv(cfg.datasets.val_data_dir)
        test_data = read_csv(cfg.datasets.test_data_dir)

        if self.mode == 'train':
            self.data = train_data
        elif self.mode == 'test':
            self.data = test_data
        elif self.mode == 'val':
            self.data = val_data

        elif self.mode == 'full':
            self.data = train_data + test_data + val_data

    def load_image_and_mask(self, img_path, mask_path):
        img = self.loader(img_path)
        mask = self.loader(mask_path)

        return img, mask

    def post_process(self, img, mask):
        img = img.float()
        mask = mask.float()
        return img, mask

    def __getitem__(self, idx):
        # load image and mask
        img_path, seg_path = self.data[idx]
        img, mask = self.load_image_and_mask(img_path, seg_path)

        # transform
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img_path, img, mask

    def __len__(self):
        return len(self.data)