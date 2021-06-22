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
        self.num_removed = 0

class Network:
    def __init__(self, settings, N):
        self.settings = settings
        self.N = N

        self.model = Sequential()
        ik = initializers.RandomNormal(stddev=0.1)
        ib = initializers.RandomNormal(stddev=0.1)

        W = 1

        self.model.add(Dense(W*self.N, input_dim=self.N, kernel_initializer=ik, bias_initializer=ib))
        self.model.add(ReLU(negative_slope=0.3))

        self.model.add(Dense(W*self.N, kernel_initializer=ik, bias_initializer=ib))
        self.model.add(ReLU(negative_slope=0.3))

        self.model.add(Dense(W*self.N, kernel_initializer=ik, bias_initializer=ib))
        self.model.add(ReLU(negative_slope=0.3))

        self.model.add(Dense(3, kernel_initializer=ik, bias_initializer=ib))
        self.model.add(ReLU(negative_slope=1.0))

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
        remove_indices = []
        idx = -1
        for ex in self.training_experience:
            idx = idx + 1
            s0 = ex.s0
            s1 = ex.s1

            I = 1.0 #math.pow(gamma, ex.step_in_ep)

            v0 = 0.0

            with tf.GradientTape() as tape:
                v0 = self.model(s0)[0][2]
            trainable = self.model.trainable_variables
            gradient_critic = tape.gradient(v0, trainable)

            v1 = self.model(s1)[0][2]
            d = ex.r1 + gamma * v1 - v0
            
            step = d * alpha * I

            weights = self.model.get_weights()
            for i in range(0, len(gradient_critic)):
                dw_critic = step * gradient_critic[i]
                weights[i] = weights[i] + dw_critic
            self.model.set_weights(weights)

            dv = v0 -self.model(s0)[0][2]

            
            print(f"v0={float(v0)}, v1={float(v1)}")
            print(f"d={d}, step={float(step)}")


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

            a1 = self.model(s0)[0][0:1]
            da = np.linalg.norm(a1, 2)
            print(f"dv={dv}, da={da}")

            if (math.fabs(dv) + math.fabs(da)) / alpha < 1:
                remove_indices.append(idx)

            if tf.math.is_nan(v0) or tf.math.is_nan(v1):
                exit()
            
            #self.model.set_weights(self.model.get_weights() + dw_critic)
        
        # Remove elements with no impact on learning
        remove_indices.reverse()
        for ri in remove_indices:
            self.training_experience.pop(ri)
        stat.num_removed = len(remove_indices)
        
        return stat


