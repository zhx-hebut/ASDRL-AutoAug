import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
# from IPython.display import clear_output
from matplotlib import pylab as plt
from RL_environment.models.replay_buffer import ReplayBuffer
from RL_environment.models.replay_buffer import NaivePrioritizedBuffer
from RL_environment.features import extract_feature
# from utils.features import extractor, batch_extractor


class Net(nn.Module):
    def __init__(self, input_shape, num_actions, device):
        super(Net, self).__init__()
        self.input_shape = input_shape 
        self.num_actions = num_actions 
        self.device = device
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=256, kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.adv = nn.Sequential(
            nn.Linear(in_features=self.feature_size(), out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.num_actions)
        )
        self.val = nn.Sequential(
            nn.Linear(in_features=self.feature_size(), out_features=512), 
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

    def feature_size(self):   
        return self.features(torch.zeros(1, *(self.input_shape, 10, 10))).view(1, -1).size(1)  

    def forward(self, x):
        x = self.features(x)  
        # print('feature size', x.size())
        x = x.view(x.size(0), -1)
        adv = self.adv(x)
        val = self.val(x).expand(x.size(0), self.num_actions)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)  
        return x

    def choose_action(self, input, epsilon):
        if random.random() > epsilon:
            state = extract_feature(input, self.device)  
            with torch.no_grad():
                # state = Variable(state.unsqueeze(0))
                q_value = self.forward(state) 
                action = q_value.max(1)[1].data[0]       
                action = action.cpu().numpy()
        else:
            action = random.randrange(self.num_actions)
        return action


class Dueling_DQN(nn.Module):
    def __init__(self, cfg):
        super(Dueling_DQN, self).__init__()
        self.num_actions = len(cfg.Environment.action_space) 
        self.device = torch.device("cuda:{}".format(cfg.agent.gpus))
        self.gamma = cfg.agent.gamma 
        self.LR = cfg.agent.lr 
        self.batch_size = cfg.agent.batch_size 
        self.replay_initial = cfg.agent.replay_initial 
        self.current_model = Net(1024, self.num_actions, self.device).to(self.device)
        self.target_model = Net(1024, self.num_actions, self.device).to(self.device)

        self.optimizer = optim.RMSprop(self.current_model.parameters(), lr=self.LR, alpha=0.95, eps=0.01)
        self.replay_buffer = NaivePrioritizedBuffer(self.replay_initial)
                
        # self.positive_action_replay_buffer = []
        # for i in range(self.num_actions):
        #     tmp = ReplayBuffer(int(self.replay_initial/10))
        #     self.positive_action_replay_buffer.append(tmp)
        # self.negative_action_replay_buffer = []
        # for i in range(self.num_actions):
        #     tmp = ReplayBuffer(int(self.replay_initial/10))
        #     self.negative_action_replay_buffer.append(tmp)

    def update_target(self): #*-*
        self.target_model.load_state_dict(self.current_model.state_dict())

    def save_DQN(self, current_model_path):
        torch.save(self.current_model.state_dict(), current_model_path)

    # def compute_td_error(self):
    #     state, action, reward, next_state, done = self.sample(self.batch_size)
    #     if state.ndim < 4: 
    #         state = np.expand_dims(state, 1)
    #         next_state = np.expand_dims(next_state, 1)
    #     action = action.astype(np.uint8)
    #     done = done.astype(np.uint8)

    #     state       = torch.from_numpy(state). float()  
    #     next_state  = torch.from_numpy(next_state).float()
    #     action      = torch.from_numpy(action).long().to(self.device)
    #     reward      = torch.from_numpy(reward).float().to(self.device)
    #     done        = torch.from_numpy(done).float().to(self.device)
    #     '''
    #     Use Unet to extract feature map
    #     '''
    #     state = extract_feature(state, self.device)
    #     next_state = extract_feature(next_state, self.device)

    #     q_values = self.current_model(state)
    #     next_q_values = self.current_model(next_state)
    #     next_q_state_values = self.target_model(next_state)

    #     q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    #     next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    #     expected_q_value = reward + self.gamma * next_q_value * (1 - done)

    #     loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss

    def compute_td_error(self, beta):
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(self.batch_size, beta)
        if state.ndim < 4: 
            state = np.expand_dims(state, 1)
            next_state = np.expand_dims(next_state, 1)

        action      = np.array(action)
        reward      = np.array(reward)
        done        = np.array(done)
        weights     = np.array(weights)
        state       = torch.from_numpy(state).float()  
        next_state  = torch.from_numpy(next_state).float()
        action      = torch.from_numpy(action).long().to(self.device)
        reward      = torch.from_numpy(reward).float().to(self.device)
        done        = torch.from_numpy(done).float().to(self.device)
        weights     = torch.from_numpy(weights).float().to(self.device)

        # state      = Variable(torch.FloatTensor(np.float32(state)))
        # next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        # action     = Variable(torch.LongTensor(action)).to(self.device)
        # reward     = Variable(torch.FloatTensor(reward)).to(self.device)
        # done       = Variable(torch.FloatTensor(done)).to(self.device)
        # weights    = Variable(torch.FloatTensor(weights)).to(self.device)
        '''
        Use Unet to extract feature map
        '''
        state = extract_feature(state, self.device)
        next_state = extract_feature(next_state, self.device)

        q_values = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices,prios.data.cpu().numpy())
        self.optimizer.step()

        return loss
    def sample(self, batch_size):
        '''
        Sample one batch size data from improved replay buffer
        :param batch_size: the batch size of dqn
        :return:
        '''
        half_batch_size = int(batch_size / 2)
        positive_state, positive_action, positive_reward, positive_next_state, positive_done = self.sample_half_batch(
            self.positive_action_replay_buffer, half_batch_size)
        negative_state, negative_action, negative_reward, negative_next_state, negative_done = self.sample_half_batch(
            self.negative_action_replay_buffer, half_batch_size)

        state = np.vstack((positive_state, negative_next_state))
        action = np.concatenate((positive_action, negative_action), axis=0)
        reward = np.concatenate((positive_reward, negative_reward), axis=0)
        next_state = np.vstack((positive_next_state, negative_next_state))
        done = np.concatenate((positive_done, negative_done), axis=0)
        # shuffle
        perm0 = np.arange(batch_size)
        np.random.shuffle(perm0)
        return state[perm0], action[perm0], reward[perm0], next_state[perm0], done[perm0]

    def sample_half_batch(self, replay_buffer, batch_size):
        average_action_size = int(batch_size / self.num_actions)
        left_action_size = batch_size - average_action_size * (self.num_actions - 1)
        min_load = 10001
        for i in range(self.num_actions):
            if min_load > len(replay_buffer[i]):
                min_load = len(replay_buffer[i])
        if min_load < average_action_size:
            assert ValueError('Min load is smaller than the size of replay buffer')
        state, action, reward, next_state, done = replay_buffer[self.num_actions - 1].sample(left_action_size)

        for i in range(self.num_actions - 1):
            tmp_state, tmp_action, tmp_reward, tmp_next_state, tmp_done = replay_buffer[i].sample(
                average_action_size)
            state = np.vstack((state, tmp_state))
            action = np.concatenate((action, tmp_action), axis=0)
            reward = np.concatenate((reward, tmp_reward), axis=0)
            next_state = np.vstack((next_state, tmp_next_state))
            done = np.concatenate((done, tmp_done))
        return state, action, reward, next_state, done

    def store(self, state, action, reward, next_state, done):
        # if reward > 0:
        #     self.positive_action_replay_buffer[action].push(state, action, reward, next_state, done)
        # else:
        #     self.negative_action_replay_buffer[action].push(state, action, reward, next_state, done)
        self.replay_buffer.push(state, action, reward, next_state, done)
