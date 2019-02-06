# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 20:46:01 2019

@author: Sergio
"""

import gym
import sys

def run_gym_enviroment(argv):
    ## El primer parametro de argv sera el nombre del entorno a ejecutar
    environment = gym.make(argv[1])
    environment.reset()
    for _ in range(int(argv[2])):
        environment.render()
        environment.step(environment.action_space.sample())
    environment.close()
    
if __name__ == "__main__":
    run_gym_enviroment(sys.argv)