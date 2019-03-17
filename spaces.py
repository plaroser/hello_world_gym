# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 12:43:51 2019

@author: Sergio
"""

import gym
from gym.spaces import *
import sys

def print_spaces(space):
    print(space)
    if isinstance(space, Box):
        print("\n Cota inferior: ",space.low)
        print("\n Cota superior: ",space.high)

if __name__ == "__main__":
    environment = gym.make(sys.argv[1])
    print("Espacio de observacionses:")
    print_spaces(environment.observation_space)
    print("espacio de acciones: ")
    print_spaces(environment.action_space)
    
    try:
        print("Descriptión de las acciones: ", environment.unwrapped.get_action_meanings())
    except AttributeError:
        pass
    