# -*- coding: utf-8 -*-
import numpy as np
import random
import tensorflow as tf
import collections
from tensorflow.keras.layers import Dense

class DQN:
    def __init__(self, num_actions, num_observations, replay_buffer_size, gamma):
        model = tf.keras.Sequential()
        model.add(Dense(64, activation='relu', input_shape=(num_observations,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_actions, activation='linear'))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

        self.build_model   = model
        self.target_model  = tf.keras.models.clone_model(self.build_model)
        self.update_target_model()
        self.replay_buffer = collections.deque(maxlen=replay_buffer_size)
        
        self.num_actions = num_actions
        self.num_observations = num_observations
        
        self.steps = 0
        self.epochs = 10
        self.sample_size = 10
        self.gamma = 0.99
    def update_target_model(self):
         self.target_model.set_weights(self.build_model.get_weights()) 
        
    
    def add_step_to_buffer(self, s, a, r, sprime, done):
        entry = dict()
        entry['s'] = s
        entry['a'] = a
        entry['r'] = r
        entry['done'] = done
        entry['sprime'] = sprime
        self.replay_buffer.append(entry)

     
    def pick_action(self, s, eps):
        x = random.uniform(0, 1)
    
        if (x < eps):
            return random.randint(0, self.num_actions-1)
        
        v = self.build_model.predict(s)
        
        return int(np.argmax(np.array(v)))
    
    
    def step(self, environment, eps):
        sraw = np.array(environment.state)
        s = np.zeros((1, sraw.shape[0]))
        s[0, :] = sraw
        a = self.pick_action(s, eps)
        sprime, r, done, info = environment.step(a)
        self.add_step_to_buffer(s, a, r, sprime, done)
        self.steps = self.steps+1
        if len(self.replay_buffer) >= 10*self.sample_size:
            sample = random.sample(self.replay_buffer, self.sample_size)
            
            xhat = np.zeros((self.sample_size, self.num_observations))
            y = np.zeros(self.sample_size)
            
            for ind in range(0, len(sample)):
                xhat[ind, :] = sample[ind]['sprime']
            
           
            yhat = self.target_model.predict(xhat)
                
            for ind in range(0, len(sample)):
                    if not sample[ind]['done']:
                        y[ind] = np.max(yhat[ind,:])*self.gamma+sample[ind]['r']
                    else:
                        y[ind] = sample[ind]['r']
            for ind in range(0, len(sample)):
                with tf.GradientTape() as tape:
                    z = self.build_model(sample[ind]['s'])
                    p = z[0, sample[ind]['a']]
                    loss = (p-y[ind])**2
            
                grad = tape.gradient(loss, self.build_model.trainable_variables)
                
                self.build_model.optimizer.apply_gradients(zip(grad, self.build_model.trainable_variables))
        
        return done
       
    def step_no_train(self, environment):
        sraw = np.array(environment.state)
        s = np.zeros((1, sraw.shape[0]))
        s[0, :] = sraw
        a = self.pick_action(s, 0)
        sprime, r, done, info = environment.step(a)
        return done