# -*- coding: utf-8 -*-

import gym
import numpy as np
import DQN_learner

environment = gym.make('MountainCar-v0')
environment.reset()
dqn = DQN_learner.DQN(environment.action_space.n, environment.observation_space.shape[0], 2000, 0.99)

steps_since_last_target_update = 0
episodes = 1000


eps = 0.1
eps_decay = 0.998
for i in range(0, episodes):
    print("EPISODE "+str(i))
    done = False
    cnt = 1
    while not done:
        done, total_reward = dqn.step(environment, eps)
        cnt = cnt+1
    eps = eps*eps_decay
    print(total_reward)
    steps_since_last_target_update = steps_since_last_target_update+cnt        
    environment.reset()
    dqn.reset()
    if steps_since_last_target_update > 100:
        #print("Updating target model!")
        steps_since_last_target_update = 0
        dqn.update_target_model()

    

    if i % 20 == 0 and i > 0:
        print("VALIDATING")
        validation_episodes = 100
        total = 0
        for i in range(0, validation_episodes):
            done = False
            while not done:
                done, total_reward = dqn.step(environment, eps)
            print(total_reward)
            total = total+total_reward
            environment.reset()
            dqn.reset()
            
        print(total/validation_episodes)
        
        if total/validation_episodes > -110:
            break