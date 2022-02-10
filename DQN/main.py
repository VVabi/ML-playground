# -*- coding: utf-8 -*-

import gym
import numpy as np
import DQN_learner

environment = gym.make('CartPole-v0')
environment.reset()
dqn = DQN_learner.DQN(environment.action_space.n, environment.observation_space.shape[0], 2000, 0.99)

steps_since_last_target_update = 0
episodes = 1000


start_eps = 0.1
for i in range(0, episodes):
    print("EPISODE "+str(i))
    cnt = 0
    while not dqn.step(environment, start_eps*(1.0-i/episodes)):
        cnt = cnt+1
    
    print(cnt)
    steps_since_last_target_update = steps_since_last_target_update+cnt        
    environment.reset()
    if steps_since_last_target_update > 100:
        print("Updating target model!")
        steps_since_last_target_update = 0
        dqn.update_target_model()

    

    if i % 20 == 0:
        cnt = 0
        print("VALIDATING")
        while not dqn.step_no_train(environment):
            cnt = cnt+1
        
        print(cnt)
        environment.reset()