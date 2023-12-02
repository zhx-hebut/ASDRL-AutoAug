import torch
import time
import numpy as np
from torch.autograd import Variable
from engine.checkpointer import remove_modules_for_DataParallel
import network.nets
from PIL import Image
from matplotlib import pyplot as plt
# from torchvision import transforms
from dataset import transforms
from torchvision.transforms import functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
feature_device = torch.device('cuda:{}'.format(0))


def get_feature_extractor():
    # global feature extractor
    path = '../best.pth'
    checkpoint = torch.load(path) #, map_location={'cuda:0':'cuda:4'}
    pretrained_dict = checkpoint['model']

    pretrained_dict = remove_modules_for_DataParallel(pretrained_dict)
    # **"{'in_ch': 1, 'nclass':2}"
    net = network.nets.FlexibleUnet.__dict__['UNet'](in_ch=1, nclass=1)
    net.load_state_dict(pretrained_dict)
    print("Loaded model from {}".format(path))
    return net


pretrained_model = get_feature_extractor()
pretrained_model.eval()
pretrained_model.to(feature_device)

def extract_feature(data, device=None):
    '''
    :param data: tensor(b, C, H, W) or numpy(C,H,W) or PIL(C,H,W)
    :param device:
    :return:
    '''
    if not torch.is_tensor(data):
        # data = F.to_tensor(data).type(torch.float32) 
        # data, _ =transforms.normalize(data, data)
        f = transforms.normalize()
        data, _ = f(data, data)
        data = data.unsqueeze(0).type(torch.FloatTensor)
    data = data.to(feature_device)
    with torch.no_grad():
        feature = pretrained_model.extract_features(data)
    return feature.to(device)