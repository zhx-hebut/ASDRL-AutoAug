import warnings

from PIL import Image
from torch.utils.data.dataset import Dataset

from dataset.transforms import to_tensor, normalize

warnings.filterwarnings('ignore')

__all__ = ['BaseDataSet']


class BaseDataSet(Dataset):
    """
    transform : a function to be applied to img and mask
    """

    def __init__(self, filenames, transform=None, mode='train', as_gray=False):
        super(BaseDataSet, self).__init__()
        self.filenames = filenames
        self.transforms = transform
        self.as_gray = as_gray
        self.mode = mode.lower()
        assert self.mode in ['train', 'test', 'val']

    def load_img(self, img_filename, mask_filename):
        """
        Load PIL Image from filename
        The output must be **PIL object** !!
        :param img_filename:
        :param is_img:
        :return:
        """
        try:
            img = Image.open(img_filename)
            mask = Image.open(mask_filename)
        except Exception as e:
            print('Error opening : {} for {}'.format(mask_filename, img_filename))

        if self.as_gray:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        mask = mask.convert('L')
        return img, mask

    def post_process(self, img, mask):
        """ process tensor image and mask after completing transformations
        :param mode:
        :param img:
        :param mask:
        :return:
        """
        return img, mask

    def __getitem__(self, index):
        # img_filename, mask_filename = self.filenames[index].split('\t')
        img_filename, mask_filename = self.filenames[index]
        funcs = [self.load_img, self.transforms, to_tensor(), normalize(), self.post_process]

        img = img_filename
        mask = mask_filename
        for f in funcs:
            if f is None:
                continue
            img, mask = f(img, mask)
        return img_filename, img, mask

    def __len__(self):
        return len(self.filenames)
