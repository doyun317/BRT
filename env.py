import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dataset import CoinDataset
import matplotlib.pyplot as plt

class env():
    def __init__(self, learning_length, batch_iter, file_name): # l_l:학습할 길이(총 시간), b_i:학습할 길이로 자른 배치의 순서 선택
        self.learning_length = learning_length
        self.coindata = CoinDataset(file_name)
        self.data = DataLoader(CoinDataset(file_name), batch_size=self.learning_length)

        for batch_idx, samples in enumerate(self.data):
            if batch_idx == batch_iter:
                break

        self.state_data = samples[0]
        self.price_data = samples[1]

        plt.plot(self.price_data)
        plt.savefig('price_{}.png'.format(batch_iter))

        self.state_data = self.state_data.squeeze()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.total_profit = torch.FloatTensor([0]).to(self.device)
        self.reward = torch.FloatTensor([0]).to(self.device)
        self.price_diff = torch.FloatTensor([0]).to(self.device)
        self.num_diff = torch.FloatTensor([0]).to(self.device)
        self.num = torch.FloatTensor([0]).to(self.device)
        self.action = torch.FloatTensor([0]).to(self.device)

    def reset(self):
        self.total_profit = torch.FloatTensor([0]).to(self.device)

        return self.state_data[0]

    # @torch.no_grad()
    def step(self, action, q_value, L_num, prev_num, curr_step):

        self.action = action
        done = False

        self.price_diff = self.price_data[curr_step] - self.price_data[curr_step - 1]

        # action 에따라 -,+ 바꿔줘야함 action[0,1,2]
        if self.action == 1:  # 무포지션
            self.num = 0

        else:
            if self.action == 0:  # 숏
                self.action = -1

            if self.action == 2:  # 롱
                self.action = 1

            self.num = self.action * q_value * L_num


        self.reward = self.num * self.price_diff
        self.total_profit += self.reward
        next_state = self.state_data[curr_step + 1]

        if curr_step == self.learning_length - 2:
            done = True

        return next_state, self.reward, self.total_profit, self.num, done
