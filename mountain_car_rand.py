# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:12:35 2019

@author: Sergio
"""

import gym
environment = gym.make("MountainCar-v0");
MAX_NUM_EPISODIES = 1000
STEPS_PER_EPISODE = 200

for episode in range(MAX_NUM_EPISODIES):
    done = False
    obs = environment.reset()
    total_reward = 0.0 ##Variable para guardar la recompensa total
    step = 0
    while not done:
        environment.render()
        action = environment.action_space.sample()##Accion aleatioria
        nex_state, reward, done, info = environment.step(action)
        total_reward += reward
        step += 1
        obs = nex_state
        
    print("\nEpisodio número {} finalizado con {} iteraciones. Recompensa final {}".format(episode,step+1,total_reward))
    
environment.close()


# QLearner Class
# __init__(self,environment)
# discretize(self, obs)
# get_action(self, obs)
# learn(self, obs, action, reward, net_obs)
# EPISILON_MIN: vamos aprendiendo mientras el incremento de aprendizaje supera dicho valor
# MAX_NUM_EPISODIES: número máximo de iteraciones que estamos dispuestos a realizar
# STEPS_PER_EPISODE: Número máximo de pasos a realizar por episodio
# ALPHA: ratio aprendizaje del agente
# GAMMA: factor de descuento del agente
# NUM_DISCRETE_BINS: número de divisiones en el caso de discretizar el espacio continuo.
EPSILON_NIM = 0.005
max_num_steps = MAX_NUM_EPISODIES * STEPS_PER_EPISODE
EPSION_DECAY = 500 * EPSILON_NIM / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

import numpy as np
class QLearner(object):
    def  __init__(self,environment):
        self.obs_shape = environment.observation_space.shape
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.obs_width = (self.obs_high - self.obs_low)/self.obs_bins
        
        self.action_shape = environment.action_space.n
        self.Q = np.zeros((self.obs_bins+1,self.obs_bins+1, self.action_shape)) # matriz de 31 x 31 x 3
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0  
        
    def discretize(self, obs):
        return tuple(((obs-self.obs_low)/self.bin_width).astype(int))
        
    def get_action(self, obs):
        discrete_obs = self.discretize(obs)
        # Seleccion de la accion en base a Epsion-Greedy
        if self.epsilon > EPSILON_NIM:
            self.epsilon -= EPSION_DECAY
        if np.random.random() > self.epsilon: # Con probabilidad 1-epsilon, elegimos la mejor posible
            return np.argmax(self.Q[discrete_obs])
        else:
            return np.choose([a for a in range(self.action_shape)]) # Con probabilidad epsilon, elegimos una al azar
        
    def learn(self, obs, action, reward, net_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(net_obs)
        td_target = reward = self.gamma * np.max(self.Q[discrete_next_obs])
        td_error = td_target - self.Q[discrete_obs][action]
        self.Q[discrete_obs][action] += self.alpha * td_error