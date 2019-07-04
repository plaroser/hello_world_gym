# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:55:11 2019

@author: Sergio
"""

import gym
import numpy as np


# EPISILON_MIN: vamos aprendiendo mientras el incremento de aprendizaje supera dicho valor
# MAX_NUM_EPISODIES: número máximo de iteraciones que estamos dispuestos a realizar
# STEPS_PER_EPISODE: Número máximo de pasos a realizar por episodio
# ALPHA: ratio aprendizaje del agente
# GAMMA: factor de descuento del agente
# NUM_DISCRETE_BINS: número de divisiones en el caso de discretizar el espacio continuo.
MAX_NUM_EPISODIES = 50000
STEPS_PER_EPISODE = 200
EPSILON_NIM = 0.005
max_num_steps = MAX_NUM_EPISODIES * STEPS_PER_EPISODE
EPSION_DECAY = 500 * EPSILON_NIM / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

class QLearner(object):
    def  __init__(self,environment):
        self.obs_shape = environment.observation_space.shape
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high - self.obs_low)/self.obs_bins
        
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
            return np.random.choice([a for a in range(self.action_shape)]) # Con probabilidad epsilon, elegimos una al azar
        
    def learn(self, obs, action, reward, net_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(net_obs)
        td_target = reward + self.gamma * np.max(self.Q[discrete_next_obs])
        td_error = td_target - self.Q[discrete_obs][action]
        self.Q[discrete_obs][action] += self.alpha * td_error

## Metodo para entrenar a nuestro agente
def train(agent, environment):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODIES):
        done = False
        obs = environment.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs) # Acción elegida segun la equacion de Q-Learning
            next_obs, reward, done, info = environment.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        
        if total_reward > best_reward:
            best_reward = total_reward
        
        print("Episodio número {} con recompensa: {}, mejor recompensa: {}, epsilon: {}".format(episode, total_reward, best_reward, agent.epsilon))
    
    ## De todas las politicas de entrenamiento que hemos obtenido devolvemos la mejor de todas
    return np.argmax(agent.Q, axis = 2)

def test(agent, environment, policy):
    done = False
    obs = environment.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)] # acción que dictamina la politica que hemos entrenado
        next_obs, reward, done, info = environment.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    environment = gym.make("MountainCar-v0")
    agent = QLearner(environment)
    learned_policy = train(agent, environment)
    monitor_path = "./monitor_output"
    environment = gym.wrappers.Monitor(environment, monitor_path, force=True)
    for _ in range(1000):
        test(agent, environment, learned_policy)
    environment.close()