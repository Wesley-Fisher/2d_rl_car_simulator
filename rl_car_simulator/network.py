from copy import Error
import math

from numpy.core.fromnumeric import clip
from rl_car_simulator.settings import CONSTANTS
from rl_car_simulator.utilities import Utility
import numpy as np
import pickle as pk
import os
from os import listdir
from os.path import isfile, join
import statistics
import time

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
from tensorflow.python.keras.backend import count_params, set_session

from .utilities import Utility
from .network_model import MyReLUModel, NetworkInputs, NetworkOutputs



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


SIG = CONSTANTS.sigma
WIDTH = Utility().normal_int_width(SIG)
GAUSS_FRAC = 1.0 / (SIG * math.sqrt(2*math.pi))

class Network:
    def __init__(self, settings, N):
        self.settings = settings
        self.N = N
        self.util = Utility()
        self.freezing = False

        # MODEL
        self._model = MyReLUModel(self.settings, N, 'model')
        self.frozen_model = MyReLUModel(self.settings, N, 'frozen')

        '''
        print("network in layers")
        print(self.input)
        print(self.target_prediction)
        print(self.advantage)
        '''
        self.session = None
        self.graph = None

        self.new_training_experiences = []
        self.training_experience = []

        #if self.settings.memory.load_saved_network:
        #    self.load_state()

        self.freeze()
        
    def get(self, x):
        while self.freezing:
            time.sleep(0.001)
        return self.model(x, self.frozen_model)

    def model(self, state, model=None):
        data = self._model.make_dummy_data()
        data.state = state

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
        
        if model is None:
            pred = self._model.predict([data])
        else:
            pred = model.predict([data])

        return pred
    
    def make_dummy_data(self):
        return self._model.make_dummy_data()

    def predict_advantage(self, ex):
        data, original = self.build_epoch_targets([ex])
        return data[0].advantage[0]

    def freeze(self):
        self.freezing = True
        self.frozen_model.copy_weights(self._model.get_weights())
        self.freezing = False
    
    def add_experience(self, exp):
        self.new_training_experiences = self.new_training_experiences + exp

    def build_epoch_targets(self, exp):
        return self._model.build_epoch_targets(exp)


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
        #print("Adding %d new samples to %d existing samples" % (len(new_exp), len(self.training_experience)))
        self.training_experience = self.training_experience + new_exp

    def fit_model(self, data, verbose=0):
        self._model.fit(data, verbose=verbose, batch_size=1)

    def train_epoch(self):

        if self.settings.debug.profile_network:
            profiler = cProfile.Profile(subcalls=False)
            profiler.enable()

        remove_indices = []
        idx = -1
        sample_results = []

        data, original = self.build_epoch_targets(self.training_experience)

        self.fit_model(data)

        new = [self.model(dat.state) for dat in data]

        for orig, dat, newP in zip(original[0:5], data[0:5], new[0:5]):
            ret = dat.ret
            target = dat.target
            adv = dat.advantage
            sOrig = "orig(%f, %f,%f)" % (orig.force.action, orig.angle.action, orig.value)
            sRet = "ret(%f)" % (ret[0])
            sTarget = "target(%f,%f,%f)" % (target[0], target[1], target[2])
            sAdv = "adv(%f)" % (adv[0])
            sNew = "new(%f,%f,%f)" % (newP.force.action, newP.angle.action, newP.value)
            print("%s->%s->%s->%s->%s" % (sOrig, sRet, sTarget, sAdv, sNew))

        sample_results= []
        for orig, newP in zip(original, new):
            results = SampleTrainingResults()
            results.c_step = float(newP.value - orig.value)
            results.af_step = float(newP.force.action - orig.force.action)
            results.aa_step = float(newP.angle.action - orig.angle.action)
            sample_results.append(results)

            bad = False
            bad = bad or  math.isnan(float(newP.value))
            bad = bad or  math.isnan(float(newP.force.action))
            bad = bad or  math.isnan(float(newP.angle.action))

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
            elif self.settings.memory.max_sample_uses > 0 and \
                 self.training_experience[i].num_uses > self.settings.memory.max_sample_uses:
                 remove_indices.append(i)
            self.training_experience[i].num_uses = self.training_experience[i].num_uses + 1
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
        self._model.save_model(network_file)

    def load_state(self):
        if self.settings._files.root_dir is None:
            return

        net = self.load_model()
        exp = self.load_experience()
        self.save_state()
        return net and exp


    def load_model(self):
        self.freezing = True # Set to False if returning
        memory_dir = self.settings._files.root_dir + "/memory"
        
        good = False
        if self.settings.memory.load_saved_network:
            orig_net = MyReLUModel(self.settings, self.N, 'temp')
            
            try:
                network_file = memory_dir + "/model.h5"
                self.graph = tf.compat.v1.Graph()
                self._model.load_model(network_file)
                self.frozen_model.load_model(network_file)
                print("Loaded Network")
                good =  True
            except OSError as e:
                print("Could not load network from file %s: %s" % (network_file, str(e)))
                good =  False
            except ValueError as e:
                print("Network shape likely mismatched: " + str(e))
                good =  False
            except Error as e:
                print("Unexpected error: %s" + str(e))
                self.freezing = False
                raise e

            new_count = np.sum([K.count_params(p) for p in set(self._model._model.trainable_weights)])
            old_count = np.sum([K.count_params(p) for p in set(orig_net._model.trainable_weights)])
            if new_count != old_count:
                print("Network shaped changed from loading")
                good = False

            if not good:
                # Failed to load. Re-Create
                self._model = MyReLUModel(self.settings, self.N, 'model')
                self.frozen_model = MyReLUModel(self.settings, self.N, 'frozen')
                self.freeze()
            else:
                self.freeze()

        self.freezing = False
        return good

    def load_experience(self):
        memory_dir = self.settings._files.root_dir + "/memory"

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
                return True
            except OSError as e:
                print("Could not prepare to load memory from file %s: %s" % (main_exp_file, str(e)))
                return False
        
        self.training_experience = []
        for file in files:
            try:
                with open(memory_dir + "/" + file, 'rb') as handle:
                    exp = pk.load(handle)
                    self.training_experience = self.training_experience + exp
            except OSError as e:
                print("Could not load memory from file %s: %s" % (main_exp_file, str(e)))
                return False
        print("Loaded %d training samples" % len(self.training_experience))

        if self.settings.memory.purge_merged_experience:
            for file in del_files:
                os.remove(memory_dir + "/" + file)
        return True
        
