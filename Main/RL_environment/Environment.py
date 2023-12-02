import gym
import logging
import random
import numpy as np
import matplotlib.pylab as plt
from RL_environment import Actions
from RL_environment.Reward import Reward


class ImageEnv(gym.Env):
    def __init__(self, cfg):
        self.__version__ = '0.3.0'
        logging.info('ImageEnv -- Version{}'.format(self.__version__))

        # Generate variables defining the environment
        self.cfg = cfg
        self.name = None
        self.init_state = None
        self.current_state = None
        self.next_state = None
        self.init_mask = None
        self.current_mask = None
        self.next_mask = None

        self.action_space = cfg.Environment.action_space 
        self.n_actions = len(self.action_space) 
        self.current_steps = 0
        self.val_steps = 0
        # self.max_steps = cfg.Environment.max_steps 
        self.done = False
        # self.init_loss, self.init_iou, self.init_dice = self._validate_net() 
        self.reward = Reward(cfg)
        self.init_dice = self.reward.init_dice

    def clear(self):                    
        self.name = None
        self.init_state = None
        self.current_state = None
        self.next_state = None
        self.init_mask = None
        self.current_mask = None
        self.next_mask = None
        self.current_steps = 0          
        self.val_steps = 0 
        self.done = False               

    def step(self, action, epoch, mode='train', epsilon=0):
        self._take_action(action)            
        reward,new_dice = self._get_reward(epsilon)          
        print('Reward:{:.5}'.format(reward)) 
        ob = self.get_next_state() 
        mask = self.get_next_mask()
        self._update(reward,epoch,mode) 
        return ob, mask, reward,new_dice, self.done, '0'

    def _get_reward(self, epsilon):
        reward,new_dice = self.reward.get_reward(self.name, self.next_state, self.next_mask, epsilon) 
        return reward,new_dice

    def _take_action(self, action):
        if action in range(0, self.n_actions): 
            action = self.action_space[action]
        else:
            assert ValueError('Action idx is invalid')

        if action == 'HF':  # horizon_flip   
            self.next_state, self.next_mask = Actions.flip(self.current_state, self.current_mask, 'h')
        elif action == 'VF':  # vertical_flip 
            self.next_state, self.next_mask = Actions.flip(self.current_state, self.current_mask, 'v')
        elif action == 'LR':  # counterclockwise rotation
            self.next_state, self.next_mask = Actions.rotation(self.current_state, self.current_mask, 5)
        elif action == 'RR':  # clockwise rotation
            self.next_state, self.next_mask = Actions.rotation(self.current_state, self.current_mask, -5)
        elif action == 'WP':  # warp
            self.next_state, self.next_mask = Actions.warp(self.current_state, self.current_mask)
        elif action == 'ZM':  # zoom_in
            self.next_state, self.next_mask = Actions.zoom(self.current_state, self.current_mask, 1.1)
        elif action == 'AN':  # add_gaussian_noise
            self.next_state, self.next_mask = Actions.add_GaussianNoise(self.current_state, self.current_mask, 0.05)
        elif action == 'LT':  # lighter
            self.next_state, self.next_mask = Actions.color(self.current_state, self.current_mask, 1., 0.075)
        elif action == 'DK':  # darker
            self.next_state, self.next_mask = Actions.color(self.current_state, self.current_mask, 1., -0.075)
        elif action == 'CL':  # crop from left
            self.next_state, self.next_mask = Actions.crop(self.current_state, self.current_mask, 'x', 20)
        elif action == 'CR':  # crop from right
            self.next_state, self.next_mask = Actions.crop(self.current_state, self.current_mask, 'x', -20)
        elif action == 'CU':  # crop from up
            self.next_state, self.next_mask = Actions.crop(self.current_state, self.current_mask, 'y', 20)
        elif action == 'CD':  # crop from ldown
            self.next_state, self.next_mask = Actions.crop(self.current_state, self.current_mask, 'y', -20)
        elif action == 'SP':  # sharpen
            self.next_state, self.next_mask = Actions.sharpen(self.current_state, self.current_mask)
        elif action == 'SM':  # smooth
            self.next_state, self.next_mask = Actions.smooth(self.current_state, self.current_mask)
        # elif action == 'TM':  # terminal
        #     self.done = True
        else:
            assert ValueError('Action idx is wrong')
        print('Action: {}  '.format(action), end=' ')

    def _update(self,reward,epoch,mode):
        self.current_state = self.next_state 
        self.current_mask = self.next_mask

        #r<0两次即停（无论训练/验证阶段）
        if reward < 0:
            self.val_steps += 1
            if self.val_steps > 1 or reward == -1: #两次连续reward<0时，停止;或者reward=-1停止，避免val时陷入+1、-1循环。
                self.done = True
        else:
            self.val_steps = 0

        #r<0一次即停（无论训练/验证阶段）
        # if reward < 0:
        #     self.done = True


    def reset(self, init_state, init_mask):
        init_mask = init_mask.astype(np.float32) 

        self.init_state = init_state 
        self.current_state = init_state
        self.next_state = init_state

        self.init_mask = init_mask  
        self.current_mask = init_mask
        self.next_mask = init_mask

        self.current_steps = 0 
        self.val_steps = 0
        self.done = False   
        self.init_dice = self.reward.reset() 

    def seed(self, seed=None):
        random.seed(seed)

    def render(self, mode='human'):
        # '''
        # show current state and label
        # :param mode:
        # :return:
        # '''
        # clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.imshow(self.current_state)
        # need to be checked
        plt.imshow(self.current_mask)
        # plt.title(str(self.label))
        plt.show()

    def get_cur_state(self):
        return self.current_state

    def get_cur_mask(self):
        return self.current_mask.astype(np.int64)

    def get_next_state(self):
        return self.next_state

    def get_next_mask(self):
        return self.next_mask.astype(np.int64)

