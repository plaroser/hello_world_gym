# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:51:36 2019

@author: Sergio
"""

from gym import envs

env_names = [env.id for env in envs.registry.all()]
for name in sorted(env_names):
    print(name)