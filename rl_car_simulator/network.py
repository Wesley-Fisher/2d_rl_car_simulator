import math
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.gen_math_ops import is_nan 


class TrainingStats:
    def __init__(self):
        self.num_samples = 0
        self.num_removed = 0


class SampleTrainingResults:
    def __init__(self):
        self.v0 = []
        self.v1 = []
        self.a = []

class Network:
    def __init__(self, settings, N):
        self.settings = settings
        self.N = N

        self.model = Sequential()
        ik = initializers.RandomNormal(stddev=0.1, seed=1)
        ib = initializers.RandomNormal(stddev=0.1, seed=2)

        W = 0.5

        WN = int(W*self.N + 1)
        self.model.add(Dense(self.N, input_dim=self.N, kernel_initializer=ik, bias_initializer=ib))
        self.model.add(ReLU(negative_slope=0.3))

        self.model.add(Dense(WN, kernel_initializer=ik, bias_initializer=ib))
        self.model.add(ReLU(negative_slope=0.3))

        self.model.add(Dense(WN, kernel_initializer=ik, bias_initializer=ib))
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

    def calculate_gradients(self, state):
        with tf.GradientTape() as tape:
            v = self.model(state)[0][2]
        trainable_critic = self.model.trainable_variables
        gradient_critic = tape.gradient(v, trainable_critic)

        with tf.GradientTape() as tape:
            a = self.model(state)[0][0:2]
        trainable_actor = self.model.trainable_variables
        gradient_actor = tape.gradient(a, trainable_actor)
        
        return v, gradient_critic, trainable_critic, a, gradient_actor, trainable_actor

    def update_weights(self, step_size, gradients, trainable):
        step_size = -step_size # For Adam. update_weights(+) -> Ascent
        optimizer = Adam(learning_rate=step_size)
        optimizer.apply_gradients(zip(gradients, trainable))
        '''
        weights = self.model.get_weights()
        for i in range(0, len(gradients)):
            dw = step_size * gradients[i]
            weights[i] = weights[i] + dw
        self.model.set_weights(weights)
        '''

    def train_sample(self, ex):
        results = SampleTrainingResults()
        s0 = ex.s0
        s1 = ex.s1
        gamma = self.settings.learning.gamma
        alpha = self.settings.learning.alpha

        I = 1.0 #math.pow(gamma, ex.step_in_ep)

        v0, gradient_critic, trainable_critic, a0, gradient_actor, trainable_actor = self.calculate_gradients(s0)
        v1 = self.model(s1)[0][2]
        if ex.next_terminal:
            v1 = 0.0

        results.v0.append(v0)
        results.v1.append(v0)
        results.a.append(a0)


        d = ex.r1 + gamma * v1 - v0
        step = d * alpha * I
        self.update_weights(float(step), gradient_critic, trainable_critic)


        sig = self.settings.statistics.sigma
        in_e = -0.5 * np.dot((ex.a0 - a0), (ex.a0 - a0) ) * sig
        density = 1.0 / (2.0 * math.pi * sig) * math.exp(in_e)
        integration_width = 2 * math.pi * sig * 0.1
        prob = integration_width * density

        # Step = alpha * delta * I * ln(grad(prob)
        #      = alpha * delta * I * grad(prob) / prob
        # Grad(prob) = d-prob/d-weights
        #            = d-prob/d-u * d-u/d-weights
        #            = d-prob/d-density * d-density/d-u * d-u/d-weights
        # d-u/d-weights <- Keras gradient update

        d_prob_wrt_density = integration_width
        mat_factor = 1.0/(sig)
        d_density_wrt_u = -0.5 * mat_factor * (ex.a0 - a0) * prob
        d_density_wrt_u = float(d_density_wrt_u[0] + d_density_wrt_u[1])

        actor_step = d * alpha * I * d_prob_wrt_density * d_density_wrt_u
        print(actor_step)
        self.update_weights(float(actor_step), gradient_actor, trainable_actor)


        v0 = self.model(s0)[0][2]
        v1 = self.model(s1)[0][2]
        a = self.model(s0)[0][0:2]
        print(a)

        results.v0.append(v0)
        results.v1.append(v0)
        results.a.append(a)
    
        return results

    def no_network_change(self, results, lim):
        da = np.linalg.norm(results.a[0] - results.a[1], 2)
        dv = results.v0[1] - results.v0[0]

        if (math.fabs(dv) + math.fabs(da)) < lim:
            return True

        return False
    
    def train_epoch(self):
        new_exp = self.new_training_experiences
        self.new_training_experiences = []
        self.training_experience = self.training_experience + new_exp

        stat = TrainingStats()
        stat.num_samples = len(self.training_experience)
        remove_indices = []
        idx = -1
        for ex in self.training_experience:
            idx = idx + 1
            results = self.train_sample(ex)

            for i in [0,1]:
                if tf.math.is_nan(results.v0[i]):
                    exit()
                if tf.math.is_nan(results.v1[i]):
                    exit()
                if tf.math.is_nan(results.a[i][0]) or tf.math.is_nan(results.a[i][1]):
                    exit()

            if self.no_network_change(results, self.settings.learning.alpha):
                remove_indices.append(i)

        # Remove elements with no impact on learning
        remove_indices.reverse()
        for ri in remove_indices:
            if ri < len(self.training_experience):
                self.training_experience.pop(ri)
        stat.num_removed = len(remove_indices)
        
        return stat


