# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym #Cargamos libreria de OpenAI Gym

envirotment = gym.make("BipedalWalker-v2") #Lanzamos una instancia del videojuego d ela montania rusa
envirotment.reset() #Limpiamos y preparamos el entorno para tomar decisione
for _ in range(2000): #Durante 2000 iteraciones
    envirotment.render() #Pintamos en pantalla la accion
    envirotment.step(envirotment.action_space.sample()) #Tomamos una decision aleatoria del conjunto disponibles
envirotment.close() #Cerramos la sesion