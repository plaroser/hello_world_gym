# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:12:35 2019

@author: Sergio
"""

import gym
enviroment = gym.make("MountainCar-v0");
MAX_NUM_EPISODIES = 1000

for episode in range(MAX_NUM_EPISODIES):
    done = False
    obs = enviroment.reset()
    total_reward = 0.0 ##Variable para guardar la recompensa total
    step = 0
    while not done:
        enviroment.render()
        action = enviroment.action_space.sample()##Accion aleatioria
        nex_state, reward, done, info = enviroment.step(action)
        total_reward += reward
        step += 1
        obs = nex_state
        
    print("\nEpisodio número {} finalizado con {} iteraciones. Recompensa final {}".format(episode,step+1,total_reward))
    
enviroment.close()