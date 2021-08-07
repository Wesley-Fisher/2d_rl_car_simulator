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

import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, ReLU, Input
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.ops.gen_math_ops import is_nan 
from tensorflow.keras import backend as K

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

        
        ik = initializers.RandomNormal(stddev=0.1, seed=1)
        ib = initializers.RandomNormal(stddev=0.1, seed=2)

        W = 0.5
        WN = int(W*self.N + 1)

        D = 2

        # https://ai.stackexchange.com/questions/18753/how-to-set-the-target-for-the-actor-in-a2c
        self.input = Input(shape=(8), name='state')
        self.target_prediction = Input(shape=(3,), name='target_in_layer')
        self.advantage = Input(shape=(1), name='advantage')
        
        '''
        print("network in layers")
        print(self.input)
        print(self.target_prediction)
        print(self.advantage)
        '''

        layer = Dense(WN, kernel_initializer=ik, bias_initializer=ib)(self.input)
        layer = ReLU(negative_slope=0.3)(layer)

        for i in range(0, D):
            layer = Dense(WN, kernel_initializer=ik, bias_initializer=ib)(layer)
            layer = ReLU(negative_slope=0.3)(layer)

        out1 = Dense(3, kernel_initializer=ik, bias_initializer=ib)(layer)
        self.out = ReLU(negative_slope=1.0)(out1)

        self._model = Model([self.input, self.target_prediction, self.advantage], self.out, name='actor_critic')

        self.frozen_model = self._model
        self.new_training_experiences = []
        self.training_experience = []

        self.compile()

    def compile(self):

        sig = self.settings.statistics.sigma
        width = self.util.normal_int_width(sig)

        gauss_fac = 1.0 / (sig * math.sqrt(2*math.pi))

        def actor_critic_loss(output, pred, advantage):
            '''
            print("in-loss printouts")
            print(output)
            print(pred)
            print(advantage)
            '''
            critic_loss = K.pow(advantage - output[0,2], 2)

            def action_loss(act, pred):
                delta = act - pred
                expo = K.square(delta/sig)
                density = gauss_fac * K.exp( expo )
                #prob = density * width
                loss = K.log(gauss_fac * width) + K.log(expo + 1e-3) * advantage
                return loss
            force_loss = action_loss(output[0,0],pred[0,0])
            angle_loss = action_loss(output[0,1],pred[0,1])

            return critic_loss + force_loss + angle_loss

        self.optimizer = Adam(learning_rate=self.settings.learning.alpha, clipnorm=1.0)
        loss = actor_critic_loss(self.out, self.target_prediction, self.advantage)
        self._model.add_loss(loss)
        self._model.compile(self.optimizer)

    def model(self, state):
        dummy_target = np.array([[1,1,1]])
        dummy_advantage = np.array([[0]])
        data = (state, dummy_target, dummy_advantage)

        '''
        print("Pred Dummies")
        print(state)
        print(state.shape)
        print(dummy_target)
        print(dummy_target.shape)
        print(dummy_advantage)
        print(dummy_advantage.shape)
        print("data")
        print(data)
        '''
        
        tens = self._model.predict(data)
        a0 = float(tens[0][0])
        a1 = float(tens[0][1])
        v = float(tens[0][2])
        return [np.array([[a0], [a1], [v]])]

    def freeze(self):
        self.frozen_model = self._model
    
    def get(self, x):
        x = x.reshape((1,self.N))
        return self.frozen_model.predict(x)
    
    def add_experience(self, exp):
        self.new_training_experiences = self.new_training_experiences + exp

    def build_epoch_targets(self, exp):
        states = []
        original = []
        targets = []
        advantages = []

        gamma = self.settings.learning.gamma

        for ex in exp:
            s0 = ex.s0
            states.append(s0)

            pred0 = self.model(s0)[0]
            pred1 = self.model(ex.s1)[0]
            original.append(pred0)

            v0 = float(pred0[2])
            v1 = float(pred1[2])

            target_critic = float(v0 + (ex.r1 + ex.G - v0))

            advantage = float(ex.r1 + gamma * v1 - v0)
            advantages.append(advantage)

            target = np.array([[ex.a_force, ex.a_angle, target_critic]])
            targets.append(target)

        return states, original, targets, advantages


    def no_network_change(self, results, lim):
        da_force = results.af_step
        da_angle = results.aa_step
        dv = results.c_step

        change = math.fabs(dv) + math.fabs(da_force) + math.fabs(da_angle)
        #print("%f < %f" % (change, lim))
        if change < lim:
            return True

        return False
    
    def add_new_experience(self):
        new_exp = self.new_training_experiences
        self.new_training_experiences = []
        self.training_experience = self.training_experience + new_exp

    def fit_model(self, states, targets, advantages, verbose=0):

        def numpify(data, size):
            N = len(data)
            data = np.array(data)
            data = data.reshape(N,size)
            #data = np.array([data[0]])
            #data = data.reshape()
            return data

        states = numpify(states, self.N)
        targets = numpify(targets, 3)
        advantages = numpify(advantages, 1)
        data = (states, targets, advantages)
        
        '''
        print("pre-fit data")
        #print(states)
        print(states.shape)
        #print(targets)
        print(targets.shape)
        #print(advantages)
        print(advantages.shape)
        #print(data)
        '''
        
         
        self._model.fit(data, verbose=verbose, batch_size=1)

    def train_epoch(self):

        if self.settings.debug.profile_network:
            profiler = cProfile.Profile(subcalls=False)
            profiler.enable()

        remove_indices = []
        idx = -1
        sample_results = []

        states, original, targets, advantages = self.build_epoch_targets(self.training_experience)

        self.fit_model(states, targets, advantages)

        new = self._model.predict(np.array(states))

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
            if self.no_network_change(result, self.settings.learning.alpha*10):
                remove_indices.append(i)
            i = i + 1
        
        remove_indices.reverse()
        
        for ri in remove_indices:
            l = len(self.training_experience)
            if ri < l and l > self.settings.memory.min_reduce_size:
                self.training_experience.pop(ri)
                num_rem = num_rem + 1
        return num_rem

    def save_state(self):
        memory_dir = self.settings._files.root_dir + "/memory"
        exp_file = memory_dir + "/experience.pk"
        with open(exp_file, 'wb') as handle:
            pk.dump(self.training_experience, handle)

        network_file = memory_dir + "/model.h5"
        #tf.keras.models.save_model(self._model, filepath=network_file)
        self._model.save(network_file)

    def load_state(self):
        memory_dir = self.settings._files.root_dir + "/memory"

        
        if self.settings.memory.load_saved_network:
            try:
                network_file = memory_dir + "/model.h5"
                self._model = keras.models.load_model(network_file)
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


        
