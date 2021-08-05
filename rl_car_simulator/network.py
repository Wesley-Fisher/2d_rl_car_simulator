import math
from rl_car_simulator.utilities import Utility
import numpy as np
import pickle as pk
import os
from os import listdir
from os.path import isfile, join
import statistics

import io
import cProfile, pstats

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam, SGD
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

        self.optimizer = Adam(learning_rate=self.settings.learning.alpha)
        self.model.compile(self.optimizer, loss='mse')

    def freeze(self):
        self.frozen_model = self.model
    
    def get(self, x):
        x = x.reshape((1,self.N))
        return self.frozen_model.predict(x)
    
    def add_experience(self, exp):
        self.new_training_experiences = self.new_training_experiences + exp

    def build_epoch_targets(self, exp):
        states = []
        original = []
        targets = []

        gamma = self.settings.learning.gamma

        for ex in exp:
            s0 = ex.s0
            states.append(s0)

            pred0 = self.model(s0)[0]
            original.append(pred0)

            v0 = pred0[2]
            s1 = ex.s1
            v1 = self.model(s1)[0][2]
            if ex.next_terminal:
                v1 = 0.0

            d = ex.r1 + gamma * v1 - v0

            target_critic = ex.r1 + gamma * v1

            def target_action(d, a, u):
                sig = self.settings.statistics.sigma
                integration_width = self.util.normal_int_width(sig)
                d_prob_wrt_density = integration_width
                d_density_wrt_u = self.util.normal_density_derivative(a, u, sig)
                step = d * d_density_wrt_u

                return a + step, step

            target_actor_force, af_step = target_action(d, ex.a_force, pred0[0])
            target_actor_angle, aa_step = target_action(d, ex.a_angle, pred0[1])

            target = np.array([[target_critic, target_actor_force, target_actor_angle]])
            targets.append(target)

        return states, original, targets


    def no_network_change(self, results, lim):
        da_force = results.af_step
        da_angle = results.aa_step
        dv = results.c_step

        if (math.fabs(dv) + math.fabs(da_force) + math.fabs(da_angle)) < lim:
            return True

        return False
    
    def add_new_experience(self):
        new_exp = self.new_training_experiences
        self.new_training_experiences = []
        self.training_experience = self.training_experience + new_exp

    def fit_model(self, states, targets):
        self.model.fit(np.array(states), np.array(targets))

    def train_epoch(self):

        if self.settings.debug.profile_network:
            profiler = cProfile.Profile(subcalls=False)
            profiler.enable()

        remove_indices = []
        idx = -1
        sample_results = []

        states, original, targets = self.build_epoch_targets(self.training_experience)

        self.fit_model(states, targets)

        new = self.model.predict(np.array(states))

        sample_results= []
        for orig, new in zip(original, new):
            results = SampleTrainingResults()
            results.c_step = float(new[0][2] - orig[2])
            results.af_step = float(new[0][0] - orig[0])
            results.aa_step = float(new[0][1] - orig[1])
            sample_results.append(results)

            bad = False
            bad = bad or  tf.math.is_nan(new[0][0])
            bad = bad or  tf.math.is_nan(new[0][1])
            bad = bad or  tf.math.is_nan(new[0][2])

            if bad:
                exit()

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
            try:
                network_file = memory_dir + "/model.h5"
                self.model = keras.models.load_model(network_file)
                print("Loaded Network")
            except OSError as e:
                print("Could not load network from file")

        if self.settings.memory.load_saved_experience:
            try:
                main_exp_file = memory_dir + "/experience.pk"
                files = []
                del_files = []
                if self.settings.memory.merge_saved_experience:
                    files = [f for f in listdir(memory_dir) if isfile(join(memory_dir, f))]
                    files = [f for f in files if f.startswith('exp') and f.endswith('.pk')]
                    del_files = [f for f in files if f != main_exp_file]
                else:
                    files = [main_exp_file]
            except OSError as e:
                print("Could not load memory from file")
        
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

        
