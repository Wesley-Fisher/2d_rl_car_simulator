import math
from rl_car_simulator.utilities import Utility
import numpy as np
import pickle as pk
import os
from os import listdir
from os.path import isfile, join
import statistics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.gen_math_ops import is_nan 

from .utilities import Utility


class EpochTrainingResults:
    def __init__(self):
        self.avg_c_step = 0.0
        self.avg_af_step = 0.0
        self.avg_aa_step = 0.0


class SampleTrainingResults:
    def __init__(self):
        self.v0 = []
        self.v1 = []
        self.a_force = []
        self.a_angle = []
        self.c_step = 0.0
        self.af_step = 0.0
        self.aa_step = 0.0
        

class Network:
    def __init__(self, settings, N):
        self.settings = settings
        self.N = N
        self.util = Utility()

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
        x = x.reshape((1,self.N))
        return self.frozen_model.predict(x)
    
    def add_experience(self, exp):
        self.new_training_experiences = self.new_training_experiences + exp

    def calculate_gradients(self, state):
        with tf.GradientTape() as tape:
            v = self.model(state)[0][2]
        trainable_critic = self.model.trainable_variables
        gradient_critic = tape.gradient(v, trainable_critic)

        with tf.GradientTape() as tape:
            a_force = self.model(state)[0][0]
        trainable_actor_force = self.model.trainable_variables
        gradient_actor_force = tape.gradient(a_force, trainable_actor_force)

        with tf.GradientTape() as tape:
            a_angle = self.model(state)[0][1]
        trainable_actor_angle = self.model.trainable_variables
        gradient_actor_angle = tape.gradient(a_angle, trainable_actor_angle)
        
        return v, gradient_critic, trainable_critic, \
               a_force, gradient_actor_force, trainable_actor_force, \
               a_angle, gradient_actor_angle, trainable_actor_angle

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

        v0, gradient_critic, trainable_critic, \
        a_force, gradient_actor_force, trainable_actor_force, \
        a_angle, gradient_actor_angle, trainable_actor_angle = self.calculate_gradients(s0)
        v1 = self.model(s1)[0][2]
        if ex.next_terminal:
            v1 = 0.0

        results.v0.append(v0)
        results.v1.append(v0)
        results.a_force.append(a_force)
        results.a_angle.append(a_angle)


        d = ex.r1 + gamma * v1 - v0
        c_step = float(d * alpha * I)
        self.update_weights(float(c_step), gradient_critic, trainable_critic)

        def update_action_weights(ex_a, a, gradients, trainables):

            sig = self.settings.statistics.sigma
            density = self.util.normal_density(ex_a, a, sig)
            integration_width = self.util.normal_int_width(sig)
            
            prob = integration_width * density
            prob_net = self.util.normal_int_prob(a, a, sig)
            imp_sample_ratio = min(2.0, max(0.5, prob_net, prob))
            if math.isnan(imp_sample_ratio):
                imp_sample_ratio = 1.0

            # Step = alpha * delta * I * ln(grad(prob)
            #      = alpha * delta * I * grad(prob) / prob
            # Grad(prob) = d-prob/d-weights
            #            = d-prob/d-u * d-u/d-weights
            #            = d-prob/d-density * d-density/d-u * d-u/d-weights
            # d-u/d-weights <- Keras gradient update

            d_prob_wrt_density = integration_width
            d_density_wrt_u = self.util.normal_density_derivative(ex_a, a, sig)

            actor_step = d * alpha * I * d_prob_wrt_density * d_density_wrt_u * imp_sample_ratio
            self.update_weights(float(actor_step), gradients, trainables)
            return float(actor_step)
    
        af_step = update_action_weights(ex.a_force, a_force, gradient_actor_force, trainable_actor_force)
        aa_step = update_action_weights(ex.a_angle, a_angle, gradient_actor_angle, trainable_actor_angle)


        v0 = self.model(s0)[0][2]
        v1 = self.model(s1)[0][2]
        a = self.model(s0)[0][0:2]
        a_force = a[0]
        a_angle = a[1]

        results.v0.append(v0)
        results.v1.append(v0)
        results.a_force.append(a_force)
        results.a_angle.append(a_angle)

        results.c_step = c_step
        results.af_step = af_step
        results.aa_tep = aa_step
    
        return results

    def no_network_change(self, results, lim):
        da_force = results.a_force[0] - results.a_force[1]
        da_angle = results.a_angle[0] - results.a_angle[1]
        dv = results.v0[1] - results.v0[0]

        if (math.fabs(dv) + math.fabs(da_force) + math.fabs(da_angle)) < lim:
            return True

        return False
    
    def add_new_experience(self):
        new_exp = self.new_training_experiences
        self.new_training_experiences = []
        self.training_experience = self.training_experience + new_exp

    def train_epoch(self):
        remove_indices = []
        idx = -1
        sample_results = []
        for ex in self.training_experience:
            idx = idx + 1
            result = self.train_sample(ex)

            for i in [0,1]:
                if tf.math.is_nan(result.v0[i]):
                    exit()
                if tf.math.is_nan(result.v1[i]):
                    exit()
                if tf.math.is_nan(result.a_force[i]):
                    exit()
                if tf.math.is_nan(result.a_angle[i]):
                    exit()
            sample_results.append(result)

        epoch_results = EpochTrainingResults()
        epoch_results.avg_c_step = statistics.mean([r.c_step for r in sample_results])
        epoch_results.avg_af_step = statistics.mean([r.af_step for r in sample_results])
        epoch_results.avg_aa_step = statistics.mean([r.aa_step for r in sample_results])

        return sample_results, epoch_results

    def remove_samples(self, training_results):
        num_rem = 0
        remove_indices = []
        i = 0
        for result in training_results:
            if self.no_network_change(result, self.settings.learning.alpha):
                remove_indices.append(i)
            i = i + 1
        
        remove_indices.reverse()
        
        for ri in remove_indices:
            l = len(self.training_experience)
            if ri < l and l < self.settings.memory.min_reduce_size:
                self.training_experience.pop(ri)
                num_rem = num_rem + 1
        return num_rem

    def save_state(self):
        memory_dir = self.settings._files.root_dir + "/memory"
        exp_file = memory_dir + "/experience.pk"
        with open(exp_file, 'wb') as handle:
            pk.dump(self.training_experience, handle)

        network_file = memory_dir + "/model.h5"
        #tf.keras.models.save_model(self.model, filepath=network_file)
        self.model.save(network_file)

    def load_state(self):
        memory_dir = self.settings._files.root_dir + "/memory"

        if self.settings.memory.load_saved_network:
            network_file = memory_dir + "/model.h5"
            self.model = keras.models.load_model(network_file)
            print("Loaded Network")

        if self.settings.memory.load_saved_experience:
            main_exp_file = memory_dir + "/experience.pk"
            files = []
            del_files = []
            if self.settings.memory.merge_saved_experience:
                files = [f for f in listdir(memory_dir) if isfile(join(memory_dir, f))]
                files = [f for f in files if f.startswith('exp') and f.endswith('.pk')]
                del_files = [f for f in files if f != main_exp_file]
            else:
                files = [main_exp_file]
        
        self.training_experience = []
        for file in files:
            with open(memory_dir + "/" + file, 'rb') as handle:
                exp = pk.load(handle)
                self.training_experience = self.training_experience + exp
        print("Loaded %d training samples" % len(self.training_experience))
        self.save_state()

        if self.settings.memory.purge_merged_experience:
            for file in del_files:
                os.remove(memory_dir + "/" + file)

        
