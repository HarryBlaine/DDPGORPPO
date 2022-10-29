import argparse
from itertools import count
from scipy.signal import savgol_filter
import os, sys, random
import numpy as np
import gym
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Portfolio_Env import PortfolioEnv
import warnings
warnings.filterwarnings("ignore")

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''
eps = 1e-8
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="Portfolio")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.01, type=float)
parser.add_argument('--max_episode', default=5001, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)

df = pd.read_csv('DAX_final(feature3).csv')

df_close = pd.DataFrame()
df_open = pd.DataFrame()
for i in range(df.shape[1]//12):
    df_close = pd.concat([df_close, df.iloc[:, i*12+3:i*12+4]], axis=1)
    df_open = pd.concat([df_open, df.iloc[:, i * 12:i * 12 + 1]], axis=1)

df_close_train = df_close.head(round(0.8 * len(df)))
df_close_test = df_close.tail(round(0.2 * len(df)))
df_open_train = df_open.head(round(0.8 * len(df)))
df_open_test = df_open.tail(round(0.2 * len(df)))

df_train = df.head(round(0.8 * len(df)))
df_test = df.tail(round(0.2 * len(df)))
train_set_obs = (df_train - np.mean(df_train.to_numpy())) / np.std(df_train.to_numpy())
test_set_obs = (df_test - np.mean(df_train.to_numpy())) / np.std(df_train.to_numpy())


train_set_obs = pd.DataFrame(train_set_obs)
test_set_obs = pd.DataFrame(test_set_obs)

env = PortfolioEnv(train_set_obs, df_close_train,df_open_train)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = df.shape[1]
action_dim = df.shape[1]//12


min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp' + script_name + args.env_name +'./'

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim*10, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = x.reshape(-1,3840)
        x = self.dropout(F.relu(self.l1(x)))
        x = self.dropout(F.relu(self.l2(x)))
        x = self.l3(x)
        x = F.softmax(x, dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.l1 = nn.Linear(state_dim*10 + action_dim, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 1)


    def forward(self, x, u):
        x = x.reshape(-1, 3840)
        x = self.dropout(F.relu(self.l1(torch.cat([x, u], 1))))
        x = self.dropout(F.relu(self.l2(x)))
        x = self.l3(x)
        return x


class DDPG(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPG, self).__init__()
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = state
        state = torch.from_numpy(state).float()
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            # print("x shape is: "+str(x.shape))
            # print("y shape is: "+str(y.shape))
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            next_state = torch.squeeze(next_state)
            state = torch.squeeze(state)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self,i):
        torch.save(self.actor.state_dict(), directory + 'actor_' + str(i) + '.pth')
        torch.save(self.critic.state_dict(), directory + 'critic_' + str(i) + '.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor_2000.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic_2000.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    agent = DDPG(state_dim, action_dim)
    agent.eval()
    total_test_reward = []
    if args.mode == 'test':
        agent.load()
        env = PortfolioEnv(train_set_obs, df_close_train, df_open_train)
        total_reward = 1
        state = env.reset(False)
        for t in count():
            action = agent.select_action(state.to_numpy())
            action /= action.sum()
            next_state, reward, done, Markowitz, info = env.step(np.float32(action))
            reward = reward / 10

            __console = sys.stdout
            f = open('action.txt', "a+")
            sys.stdout = f
            print(action)
            sys.stdout = __console
            __console = sys.stdout
            f = open('reward.txt', "a+")
            sys.stdout = f
            print(reward)
            sys.stdout = __console

            total_reward = total_reward * (1 + reward)

            state = next_state

            if done:
                print("the reward is \t{:0.2f}, the step is \t{}".format( total_reward, t))
                total_test_reward.append(total_reward)
                break


    elif args.mode == 'train':
        total_train_reward = []
        total_test_reward = []
        total_train_Markowitz = []
        total_test_Markowitz = []
        if args.load: agent.load()
        total_step = 0
        for i in range(args.max_episode):
            env_train = PortfolioEnv(train_set_obs, df_close_train,df_open_train)
            total_reward = 1
            step = 0
            state = env_train.reset(True)

            for t in count():

                action = agent.select_action(state.to_numpy())
                action = (action + np.random.normal(0, args.exploration_noise, size=(32))).clip(
                    0, 1)
                action /= action.sum()
                if t == 5:
                    __console = sys.stdout
                    f = open('action.txt', "a+")
                    sys.stdout = f
                    print(action)
                    sys.stdout = __console

                next_state, reward, done, Markowitz,info = env_train.step(action)
                reward_show = reward
                reward = reward - Markowitz/10
                if (action > 0.5).any():
                    reward = reward - 0.1

                #print("x shape is: " + str(state.shape))
                #print("y shape is: " + str(next_state.shape))
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
                state = next_state

                if done:
                    break
                step += 1
                reward_show = reward_show / 10
                total_reward = total_reward * (1+reward_show)


            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(step, i, total_reward))

            total_train_reward.append(total_reward)
            total_train_Markowitz.append(Markowitz.mean())
            agent.update()


            if i % 100 == 0:
                agent.save(i)
            if i % 1 == 0:

                env_test = PortfolioEnv(test_set_obs,df_close_test,df_open_test)
                total_reward = 1
                state = env_test.reset(False)
                for t in count():
                    state = state.to_numpy()
                    action = agent.select_action(state)
                    action /= action.sum()
                    #print(action)
                    next_state, reward, done,Markowitz, info = env_test.step(np.float32(action))
                    reward = reward / 10
                    total_reward = total_reward * (1 + reward)
                    # env.render()
                    if done:
                        print("Testing!!! , the total reward is \t{:0.2f}, the step is \t{}".format(total_reward, t))
                        total_test_reward.append(total_reward)
                        total_test_Markowitz.append(Markowitz.mean())
                        break
                    state = next_state
        plt.plot(total_train_reward,'y')
        y = savgol_filter(total_train_reward, 51, 3, mode='nearest')
        plt.plot(y, 'b')
        p1 = "./image/DAX_train_image_3"
        plt.savefig(p1)
        plt.figure()


        plt.plot(total_test_reward,'y')
        y = savgol_filter(total_test_reward, 17, 3, mode='nearest')
        plt.plot(y, 'b')
        p2 = "./image/DAX_test_image_3"
        plt.savefig(p2)
        plt.figure()

        plt.plot(total_train_Markowitz)
        p3 = "./image/DAX_train_Markowitz_image_3"
        plt.savefig(p3)

        plt.figure()

        plt.plot(total_test_Markowitz)
        p4 = "./image/DAX_test_Markowitz_image_3"
        plt.savefig(p4)
        plt.show()
    else:
        raise NameError("mode wrong!!!")



if __name__ == '__main__':
    main()