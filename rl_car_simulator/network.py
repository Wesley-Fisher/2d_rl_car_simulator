import math
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras import initializers
from tensorflow.python.ops.gen_math_ops import is_nan 


class TrainingStats:
    def __init__(self):
        self.num_samples = 0

class Network:
    def __init__(self, settings, N):
        self.settings = settings
        self.N = N

        self.model = Sequential()
        ik = initializers.RandomNormal(stddev=0.5)
        ib = initializers.RandomNormal(stddev=0.5)

        self.model.add(Dense(2*self.N, input_dim=self.N, kernel_initializer=ik, bias_initializer=ib))
        self.model.add(ReLU(negative_slope=0.3, max_value=100.0))

        self.model.add(Dense(2*self.N, kernel_initializer=ik, bias_initializer=ib))
        self.model.add(ReLU(negative_slope=0.3, max_value=100.0))

        self.model.add(Dense(2*self.N, kernel_initializer=ik, bias_initializer=ib))
        self.model.add(ReLU(negative_slope=0.3, max_value=100.0))

        self.model.add(Dense(3, kernel_initializer=ik, bias_initializer=ib))
        self.model.add(ReLU(negative_slope=1.0, max_value=100.0))

        self.frozen_model = self.model
        self.new_training_experiences = []
        self.training_experience = []

    def freeze(self):
        self.frozen_model = self.model
    
    def get(self, x):
        x = x.reshape((1,8))
        return self.frozen_model.predict(x)
    
    def add_experience(self, exp):
        self.new_training_experiences = self.new_training_experiences + exp
    
    def train_epoch(self):
        new_exp = self.new_training_experiences
        self.new_training_experiences = []
        self.training_experience = self.training_experience + new_exp

        gamma = self.settings.learning.gamma
        alpha = self.settings.learning.alpha

        stat = TrainingStats()
        stat.num_samples = len(self.training_experience)
        for ex in self.training_experience:
            s0 = ex.s0
            s1 = ex.s1

            I = math.pow(gamma, ex.step_in_ep)

            v0 = 0.0

            with tf.GradientTape() as tape:
                v0 = self.model(s0)[0][2]
            trainable = self.model.trainable_variables
            gradient_critic = tape.gradient(v0, trainable)

            v1 = self.model(s1)[0][2]
            d = ex.r1 + gamma * v1 - v0
            

            weights = self.model.get_weights()
            for i in range(0, len(gradient_critic)):
                dw_critic = d * alpha * I * gradient_critic[i]
                weights[i] = weights[i] + dw_critic
            self.model.set_weights(weights)

            dv = v0 -self.model(s0)[0][2]

            
            print("V")
            print(v0)
            print(v1)
            print(float(dv))


            a0 = 0.0
            with tf.GradientTape() as tape:
                a0 = self.model(s0)[0][0:1]
                
            trainable = self.model.trainable_variables
            gradient_actor = tape.gradient(a0, trainable)

            prob = 1.0 / 2.0 * math.pi * math.exp(-0.5 * np.dot((ex.a0 - a0), (ex.a0 - a0) ))
            
            weights = self.model.get_weights()
            for i in range(0, len(gradient_actor)):
                act_grad = prob * a0 * gradient_actor[i]
                dw_actor = d * alpha * I * act_grad
                weights[i] = weights[i] + dw_actor
            self.model.set_weights(weights)

            if tf.math.is_nan(v0) or tf.math.is_nan(v1):
                exit()
            
            #self.model.set_weights(self.model.get_weights() + dw_critic)
        return stat


