import os
import math
import time
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
import RL_environment.inits as inits
from RL_environment.Environment import ImageEnv
from RL_environment.models.DQN import Dueling_DQN
from preprocess.tools import tensor_to_image, mkdir, format_time
# import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
stage_dict = {
    'train': 2,
    'continue': 3,
    'test': 1
}


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)  

beta_start = 0.4
beta_frames = 100000
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

torch.multiprocessing.set_sharing_strategy('file_system')


class Trainer(object):
    def __init__(self, cfg):
        print('Constructing components...')
        self.cfg = cfg
        self.epochs = cfg.trainer.epochs  
        self.replay_init = cfg.trainer.replay_init 
        self.stage = cfg.trainer.stage 
        self.test_epoch = cfg.trainer.test_epoch 
        self.start_eval = cfg.trainer.start_eval 
        self.output_path = Path(cfg.trainer.output_path)
        self.output_path.mkdir(exist_ok=True)

        seed = cfg.trainer.seed 
        self.set_seed(seed)

        self.env = ImageEnv(cfg)
        self.train_loader = inits.get_dqn_train_loader(cfg)
        self.val_loader, self.test_loader = inits.get_dataloader(cfg, ['val', 'test'])
        self.dqn = Dueling_DQN(cfg)

        # components for test
        self.net = None
        self.opt = None
        self.loss_func = None
        self.eval_func = None
        self.scheduler = None

        # to record log
        self.logger = inits.get_logger(cfg)
        self.logger.info('')

        self.set_training_stage(self.stage)
        pass

    def set_training_stage(self, stage):
        stage = stage.strip().lower()
        self.stage = stage_dict[stage] 
        if stage == 'continue':
            raise NotImplementedError('Load pretrained dqn parameters')
        if stage == 'test':
            # components for test
            self.net = inits.get_dqn_network(self.cfg)
            self.opt = inits.get_solver(self.net, self.cfg)
            self.loss_func = inits.get_loss_func(self.cfg)
            self.eval_func = inits.get_eval_func(self.cfg)
            self.scheduler = inits.get_dqn_scheduler(self.opt, self.train_loader, self.cfg)

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _train_dqn(self, epoch):
        rewards = 0.
        losses = 0.
        train_nums = len(self.train_loader.dataset)
        for batch_idx, (name, scan, mask) in enumerate(self.train_loader): 
            frame_idx = epoch * train_nums + batch_idx
            epsilon = epsilon_by_frame(frame_idx) 
            beta = beta_by_frame(frame_idx)
            print('Frame idx:{}'.format(frame_idx))
            scan, mask = tensor_to_image(scan), tensor_to_image(mask) 
            self.env.reset(scan, mask) 

            state = scan
            while True:
                action = self.dqn.current_model.choose_action(state, epsilon)
                next_state, _, reward, _,done, _ = self.env.step(action, epoch, 'train', epsilon)
                self.dqn.store(state, action, reward, next_state, done)
                state = next_state
                rewards += reward 

                if frame_idx > self.replay_init: 
                    losses += self.dqn.compute_td_error(beta)

                if done:
                    if frame_idx % 30 == 0:
                        self.dqn.update_target()
                    break

        rewards /= train_nums
        losses /= train_nums
        return losses, rewards

    def _validate_dqn(self,epoch):
        with open(os.path.join(self.cfg.trainer.image_output_path,'action.txt'),'a') as log:
            log.write('{}\n'.format(epoch))
        dice_=[]
        for name, scan, mask in self.train_loader:
            scan, mask = tensor_to_image(scan), tensor_to_image(mask)
            self.env.reset(scan, mask)
            state = scan
            action_list = []
            while True:
                action = self.dqn.current_model.choose_action(state, -1.0) 
                next_state, next_mask, reward,new_dice, done, _ = self.env.step(action, epoch, 'val') 
                state = next_state
                mask = next_mask
                action_list.append(action) 

                img_name = os.path.split(name[0])[-1]
                if done: 
                    with open(os.path.join(self.cfg.trainer.image_output_path,'action.txt'),'a') as log:
                        log.write('{}:{}\n'.format(img_name,action_list))              
                    final_scan = torch.tensor(state)
                    final_mask = torch.tensor(mask.astype(np.float32))
                    dice_.append(new_dice)
                    if not os.path.exists(os.path.join(self.cfg.trainer.image_output_path, 'img')):
                        os.makedirs(os.path.join(self.cfg.trainer.image_output_path, 'img'))
                    if not os.path.exists(os.path.join(self.cfg.trainer.image_output_path, 'label')):
                        os.makedirs(os.path.join(self.cfg.trainer.image_output_path, 'label'))
                    save_image(final_scan, os.path.join(self.cfg.trainer.image_output_path, 'img', name[0].split('/')[-1].split('.')[0] + '_aug' + '.jpg'))
                    save_image(final_mask, os.path.join(self.cfg.trainer.image_output_path, 'label', name[0].split('/')[-1].split('.')[0] + '_aug' + '.png'))

                    break
        dice = sum(dice_)/len(dice_) 
        return dice
    def train_dqn(self):
        best_dice = 0
        for epoch in range(self.epochs):
            loss, reward = self._train_dqn(epoch) 
            self.logger.info('Train_Epoch:{} || episode loss:{}, episode reward:{}'.format(epoch, loss, reward))  
            # save model
            if epoch >= 1:  
                dice = self._validate_dqn(epoch)    
                if best_dice < dice:
                    best_dice = dice
                    with open(os.path.join(self.cfg.trainer.image_output_path,'Valdice.txt'),'a') as log:
                        log.write('{}:{}\n'.format(epoch,dice))
                    imgpath = os.path.join(self.cfg.trainer.image_output_path, 'img')
                    imgname = os.listdir(imgpath)
                    for i in imgname:
                        img_path = os.path.join(imgpath,i)
                        labelname = i.split('.')[0] + '.png'
                        label_path = os.path.join(self.cfg.trainer.image_output_path, 'label',labelname)
                        img = Image.open(img_path)
                        label = Image.open(label_path)
                        if not os.path.exists(os.path.join(self.cfg.trainer.image_output_bestpath, 'img')):
                            os.makedirs(os.path.join(self.cfg.trainer.image_output_bestpath, 'img'))
                        if not os.path.exists(os.path.join(self.cfg.trainer.image_output_bestpath, 'label')):
                            os.makedirs(os.path.join(self.cfg.trainer.image_output_bestpath, 'label'))
                        img.save(os.path.join(self.cfg.trainer.image_output_bestpath, 'img',i))
                        label.save(os.path.join(self.cfg.trainer.image_output_bestpath, 'label',labelname))


    def test_dqn(self):
        pass