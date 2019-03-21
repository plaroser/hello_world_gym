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