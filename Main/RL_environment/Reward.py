import torch
import numpy as np
from dataset import transforms
import torch.nn as nn
import RL_environment.inits as inits
from engine.checkpointer import remove_modules_for_DataParallel, add_modules_for_DataParallel

class Reward(object):
    def __init__(self, cfg):
        super(Reward, self).__init__()
        self.cfg = cfg
        self.init_loss = None
        self.gpus = [int(i) for i in cfg.Reward.gpus.split(',')]
        self.val_loader = inits.get_dqn_val_loader(cfg)
        self.net = inits.get_reward_network(cfg) 
        # self.device = torch.device("cuda:{}".format(cfg.Reward.gpus))
        self.device = torch.device(self.gpus[0]) 

        print('Reward gpus id:{}'.format(self.gpus))
        if len(self.gpus) > 1: 
            self.net = nn.DataParallel(self.net, device_ids=self.gpus)
        self.net.cuda(self.gpus[0])
        self.load_pretrained_params()

        self.opt = inits.get_solver(self.net, cfg) 
        self.loss_func = inits.get_loss_func(cfg) 
        self.eval_func = inits.get_eval_func(cfg) 
        # self.init_loss, _, _ = self._validate_net()
        self.init_loss, self.init_iou, self.init_dice = self._validate_net() 
        self.pre_dice = self.init_dice 
        print('Init loss:{}'.format(self.init_loss)) 
        print('Init iou:{}'.format(self.init_iou))
        print('Init dice:{}'.format(self.init_dice))

    def load_pretrained_params(self): 
        checkpoint = torch.load(self.cfg.Reward.basic_params_path) #best.pth #, map_location={'cuda:0':'cuda:4'}
        pretrained_dict = checkpoint['model']
        if not isinstance(self.net, nn.DataParallel) and 'module.' in list(pretrained_dict.keys())[0]:
            pretrained_dict = remove_modules_for_DataParallel(pretrained_dict)
        if isinstance(self.net, nn.DataParallel) and 'module.' not in list(pretrained_dict.keys())[0]:
            pretrained_dict = add_modules_for_DataParallel(pretrained_dict)
        self.net.load_state_dict(pretrained_dict)

    def reset(self):
        self.pre_dice = self.init_dice  
        checkpoint = torch.load(self.cfg.Reward.basic_params_path) #best.pth #, map_location={'cuda:0':'cuda:4'}
        pretrained_dict = checkpoint['model']
        if not isinstance(self.net, nn.DataParallel) and 'module.' in list(pretrained_dict.keys())[0]:
            pretrained_dict = remove_modules_for_DataParallel(pretrained_dict)
        if isinstance(self.net, nn.DataParallel) and 'module.' not in list(pretrained_dict.keys())[0]:
            pretrained_dict = add_modules_for_DataParallel(pretrained_dict)
        self.net.load_state_dict(pretrained_dict) 
        return self.init_dice

    def inference(self, name, input, label):
        input = input.to(self.device) 
        label = label.to(self.device)
        
        logits = self.net(input) 
        if torch.isinf(logits[0]).any():
            raise Exception("Nan when validating, data : {}".format(name))

        # calculate loss by user
        loss = self.loss_func(logits, label)  
        # get accuracy with dice_loss
        with torch.no_grad():
            self.eval_func(logits, label)
        return loss, logits

    def _validate_net(self):
        """
        Validate current model
        :param dataloader:  a  dataloader, it should be set to run in validate mode
        :return:
        """
        self.eval_func.clear_cache()
        self.loss_func.clear_cache()
        self.net.eval() 
        self.loss_func.eval()
        self.eval_func.eval()
        losses = 0.
        nTotal = 0
        for step, (name, batch_x, batch_y) in enumerate(self.val_loader): 
            # forward
            loss, logits = self.inference(name, batch_x, batch_y)
            losses += float(loss)
            nTotal += 1
        # Log
        losses /= nTotal
        iou, dice = self.eval_func.get_last() 
        return losses, iou, dice
        pass

    def _update(self, name, data, target):
        self.eval_func.clear_cache()
        self.loss_func.clear_cache()
        self.net.train() 
        self.loss_func.train()
        self.eval_func.train()
        loss, logits = self.inference(name, data, target) 
        self.opt.zero_grad() 
        loss.backward()      
        self.opt.step()      
        pass

    def get_reward(self, name, data, target, epsilon):
        '''
        :param name:
        :param data: numpy or PIL or tensor with (B,C,H,W)
        :param target:
        :return:
        '''
        if not torch.is_tensor(data):                             
            # f = transforms.to_tensor()
            f = transforms.normalize()
            data, target = f(data, target)
            data, target = data.unsqueeze(0).type(torch.FloatTensor), target.unsqueeze(0).type(torch.FloatTensor) 

        self._update(name, data, target) 
        _, _, new_dice = self._validate_net() 

        # reward = (new_dice - self.pre_dice)*100 

        #ours
        if new_dice - self.init_dice>0 and new_dice - self.pre_dice>0:
            reward = 1
        elif new_dice - self.init_dice>0 and new_dice - self.pre_dice<=0:
            reward =  (new_dice - self.pre_dice) * 100
        elif new_dice - self.init_dice<=0 and new_dice - self.pre_dice>0:
            reward = -1
        elif new_dice - self.init_dice<=0 and new_dice - self.pre_dice<=0:
            reward = -1

        reward = np.clip(reward, -1., 1.)          
        self.pre_dice = new_dice  

        return reward,new_dice
