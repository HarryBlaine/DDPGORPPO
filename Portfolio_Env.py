import random

from universal.algos import *
import gym
from universal import tools


eps = 1e-8


class PortfolioEnv(gym.Env):

    current_state = 0
    Random = False

    def __init__(self, df_obs, df_close, df_open):
        super(PortfolioEnv, self).__init__()
        self.df_open = df_open
        self.df_close = df_close
        self.df_obs = df_obs
        self.cov = df_obs.cov().to_numpy()
    # get the observation space from the data min and max
    def reset(self, rand):
        self.Random = rand
        if not rand:
            # Reset the state of the environment to an initial state
            self.current_state = 10
            self.start_state = self.current_state
        else:
            self.current_state = random.randint(10, self.df_obs.shape[0] - 520)
            self.start_state = self.current_state
        return self.give_next_observation()

    def first_day_observation(self):
        obs = self.df_open.iloc[self.current_state : self.current_state + 1, :]
        return obs

    def last_day_observation(self):
        obs = self.df_close.iloc[self.current_state + 9 : self.current_state + 10, :]
        return obs

    def give_next_observation(self):
        obs = self.df_obs.iloc[self.current_state : self.current_state + 10, :]
        return obs


    def step(self, action):
        # Execute one time step within the environment
        self.current_state = self.current_state + 10
        Markowitz = 0
        map_X = self.last_day_observation().to_numpy()/self.first_day_observation().to_numpy()
        r = (map_X - 1) * action
        for i in range(len(action)):
            for j in range(len(action)):
                Markowitz = Markowitz + action[i]*action[j]*self.cov[i, j]
        r = r.sum(axis=1) + 1
        r = r.prod() - 1
        reward = r * 10

        if self.Random:
            done = self.current_state >= self.start_state + 500
        else:
            done = self.current_state >= self.df_obs.shape[0] - 20

        obs = self.give_next_observation()  # next state

        return obs, reward, done, Markowitz, {}



