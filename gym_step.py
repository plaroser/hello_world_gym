# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 12:11:34 2019

@author: Sergio
"""

import gym #Cargamos libreria de OpenAI Gym

ENVIROMENT_NAME = "Qbert-v0"
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

envirotment = gym.make(ENVIROMENT_NAME) 

for episode in range(MAX_NUM_EPISODES):
    obs = envirotment.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        envirotment.render()
        action = envirotment.action_space.sample() #Tomamos una decion aleatoria
        next_state, reward, done, info = envirotment.step(action)
        obs = next_state
        
        if done is True:
            print("\n Episodio #{} terminado en {} steps.".format(episode, step+1))
            break
    

envirotment.close() #Cerramos la sesion