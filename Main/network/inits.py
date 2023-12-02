import os
import shutil
from torch.optim import SGD
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from network import customized_loss
from network import schedulers
import network.customized_evaluate as Eval
import network.nets
import dataset as ds
from engine.logger import *
from dataset import transforms
from preprocess.tools import format_time
from network.nets.FlexibleUnet import UNet


__all__ = ['get_network', 'get_solver', 'get_dataloader',
           'get_loss_func', 'get_eval_func', 'get_logger', 'get_scheduler']


def get_network(cfg):   
    name = cfg.model.net  
    in_ch = cfg.model.in_ch 
    nclass = cfg.model.nclass 
    args = eval(cfg.model.kwargs)
    args['in_ch'] = in_ch  
    args['nclass'] = nclass 
    if name == 'unet':
        name = 'UNet'
    elif name == 'vbnet':
        name = 'VBNet2d'
    else:
        assert ValueError('model name ought to be in [unet, vbnet]')
    # net = network.nets.FlexibleUnet.__dict__[name](**args)
    net = UNet(**args)
    # net = network.nets.UNet(**args)
    return net


def get_solver(net, cfg):
    opt_name = cfg.solver.optimizer.lower() 
    lr = cfg.solver.base_lr  
    wd = cfg.solver.weight_decay   
    if opt_name == 'adam':
        return Adam(lr=lr, params=list(net.parameters()), weight_decay=wd)
    elif opt_name == 'sgd':
        momentum = cfg.solver.momentum
        return SGD(params=list(net.parameters()), lr=lr, weight_decay=wd, momentum=momentum)
    else:
        raise Exception("Not support opt : {}".format(opt_name))


def get_scheduler(opt, dataloader, cfg):
    scheduler_name = cfg.scheduler.name.lower() 
    epoch = cfg.trainer.epoch 
    iteration = len(dataloader) 
    warmup_epochs = cfg.scheduler.warmup_epochs 
    iteration_decay = cfg.scheduler.iteration_decay 

    kwargs = eval(cfg.scheduler.kwargs)
    if not isinstance(kwargs, dict):
        raise Exception("scheduler args should be string of dict, e.g. '{k1:v1}'")

    args = {
        'optimizer': opt,
        'total_epoch': epoch,
        'iteration_per_epoch': iteration,
        'warmup_epochs': warmup_epochs,
        'iteration_decay': iteration_decay
    }
    if scheduler_name == 'poly':
        return schedulers.PolyLR(**args, **kwargs)
    if scheduler_name == 'step':
        return schedulers.StepLR(**args, **kwargs)
    if scheduler_name == 'cos':
        return schedulers.CosineLR(**args, **kwargs)


def get_loss_func(cfg) -> network.customized_loss.LoggedLoss:
    loss_func_name = cfg.evaluate.loss_func  
    nclass = cfg.datasets.nclass  
                                    
    loss_func_kwargs = eval(cfg.evaluate.loss_kwargs) 
    loss_func_kwargs['nclass'] = nclass
    if not isinstance(loss_func_kwargs, dict):
        raise Exception("Loss func args should be string of dict, e.g. '{k1:v1}'")

    if loss_func_name == 'ce':
        return customized_loss.ce_loss(**loss_func_kwargs)
    if loss_func_name == 'bce':
        return customized_loss.bce_loss(**loss_func_kwargs)
    if loss_func_name == 'dice':
        return customized_loss.dice_loss(**loss_func_kwargs)
    if loss_func_name == 'bcedice':
        return customized_loss.bcedice_loss()
    if loss_func_name == 'sdice':
        return customized_loss.simplified_dice(**loss_func_kwargs)
    if loss_func_name == 'focal':
        return customized_loss.simplified_dice(**loss_func_kwargs)
    if loss_func_name == 'mse':
        return customized_loss.mse_loss(**loss_func_kwargs)


def get_eval_func(cfg) -> Eval.SegMeasure:
    eval_func_name = cfg.evaluate.eval_func 
    eval_func_kwargs = eval(cfg.evaluate.eval_kwargs) #{'multi_inputs_axis': -1}

    if not isinstance(eval_func_kwargs, dict):
        raise Exception("Eval func args should be string of dict, e.g. '{k1:v1}'")

    if eval_func_name == 'dice' or 'iou':
        return Eval.SegMeasure(**eval_func_kwargs)
    else:
        assert ValueError('Eval name is wrong')


def get_transforms(mode):
    if mode == 'train':
        transfrom = transforms.compose([
            # transforms.resize((256, 256), (256, 256)),
            # transforms.random_crop((296, 296)),
            # transforms.random_flip(),
            transforms.normalize()
        ])
    elif mode == 'val' or 'test':
        transfrom = transforms.compose([
            transforms.normalize()
        ])
    else:
        raise ValueError('Mode is wrong')
    return transfrom


def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test', 'full', 'pred']
    dataset = None

    if mode in ['train', 'val', 'test']:
        name = cfg.datasets.name 
        transform = get_transforms(mode)
        if name == 'Kidney':
            dataset = ds.KidneyBaseDataSet(cfg, transform=transform, mode=mode)
        elif name == 'KiTS':
            dataset = ds.KiTSBaseDataSet(cfg, transform=transform, mode=mode)
        else:
            raise ValueError('DataSet loading wrong')

    elif mode == 'full':
        raise NotImplementedError('Function has not be implemented')

    elif mode == 'pred':
        raise NotImplementedError('Function has not be implemented')

    return dataset


def get_dataloader(cfg, modes):
    shuffle = cfg.dataloader.shuffle 
    drop_last = cfg.dataloader.drop_last 
    num_workers = cfg.dataloader.num_workers 
    test_batch_size = cfg.dataloader.test_batch_size 
    train_batch_size = cfg.dataloader.train_batch_size 

    loaders = []
    for mode in modes:
        dataset = get_dataset(cfg, mode)
        if mode == 'train':
            loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=False, drop_last=drop_last)
        elif mode in ['val', 'test']:
            loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=False, drop_last=drop_last)
        elif mode in ['full', 'pred']:
            if dataset is not None:
                loader = DataLoader(dataset, batch_size=1, shuffle=shuffle,
                                    num_workers=num_workers, pin_memory=False, drop_last=drop_last)
            else:
                loader = None
        loaders.append(loader)
    return loaders


def get_logger(cfg, name=None):
    logger = create_logger(cfg.datasets.name)
    cur_time = format_time()

    if 'file' in cfg.trainer.loggers:
        if name is None:
            log_name = 'log_' + cfg.model.net + '_' + cfg.trainer.stage + '_' + cur_time + '.txt'
        else:
            log_name = name + '_' + cfg.model.net + '_' + cfg.trainer.stage + '_' + cur_time + '.txt'

        log_path = os.path.join(cfg.trainer.output_path, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        # save config
        shutil.copy(cfg.cfg, log_path)
        # save logger
        add_filehandler(logger, log_path)
    return logger

