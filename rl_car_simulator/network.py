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
from tensorflow.python.keras.backend import set_session

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


SIG = CONSTANTS.sigma
WIDTH = Utility().normal_int_width(SIG)
GAUSS_FRAC = 1.0 / (SIG * math.sqrt(2*math.pi))

class NetworkInputs:
    def __init__(self):
        self.state = None
        self.target = None
        self.advantage = None
        self.ratio_force = None
        self.ratio_angle = None
        self.ret = None

class NetworkOutputs:
    def __init__(self):
        self.value = 0.0
        self.action_force = 0.0
        self.action_steer = 0.0

class MyModel:
    def __init__(self, settings,N, name):
        self.settings = settings
        self.N = N
        self.name = name
        self.util = Utility()

        self.graph = tf.Graph()
        self.session = tf.compat.v1.Session(graph=self.graph)

        self.input = None
        self.target_prediction = None
        self.advantage = None
        self.ratio_f = None
        self.ratio_a = None
        self.out = None
        self._model = None
        self.optimizer=None

        with self.graph.as_default():
            set_session(self.session)
            self.make_model()
            self.compile()

    def load_model(self, file):
        graph = tf.Graph()
        sess = tf.compat.v1.Session(graph=graph)
        try:
            with graph.as_default():
                set_session(sess)
                model = keras.models.load_model(file)
            self.session.close()
            self.graph = graph
            self.session = sess
            with self.graph.as_default():
                set_session(self.session)
                self._model = model
                self.handle_new_model()
        except OSError as e:
            raise e

    def copy_weights(self, weights):
        self.session.close()
        self.graph = tf.Graph()
        self.session = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            set_session(self.session)
            self.make_model()
            self.handle_new_model(False)
            self._model.set_weights(weights)
            self.compile()
            self.dummy_test()

    def get_weights(self):
        with self.graph.as_default():
            set_session(self.session)
            return (self._model.get_weights())

    def handle_new_model(self, compile=True):
        inputs = self._model.inputs
        self.input = inputs[0]
        self.target_prediction = inputs[1]
        self.advantage = inputs[2]
        self.ratio_f = inputs[3]
        self.ratio_a = inputs[4]
        '''
        print(self.input)
        print(self.target_prediction)
        print(self.advantage)
        '''
        self.out = self._model.output
        self._model = Model([self.input, self.target_prediction, self.advantage, self.ratio_f, self.ratio_a], self.out, name=self.name+'_actor_critic')
        if compile:
            self.compile()

    def save_model(self, filename):
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            self._model.save(filename, save_format='h5')

    def make_model(self):
        # MODEL NETWORK
        self.W = 0.5
        self.D = 2
        ik = initializers.RandomNormal(stddev=0.1, seed=1)
        ib = initializers.RandomNormal(stddev=0.1, seed=2)
        WN = int(self.W*self.N + 1)

        self.input = Input(shape=(8), name=self.name+'_state_in')
        self.target_prediction = Input(shape=(3,), name=self.name+'_target_in_layer')
        self.advantage = Input(shape=(1), name=self.name+'_advantage_in')
        self.ratio_f = Input(shape=(1), name=self.name+'_rat_f_in')
        self.ratio_a = Input(shape=(1), name=self.name+'_rat_a_in')

        layer = Dense(WN, kernel_initializer=ik, bias_initializer=ib, name="dense_input")(self.input)
        layer = ReLU(negative_slope=0.3, name="dense_input_relu")(layer)

        for i in range(0, self.D):
            layer = Dense(WN, kernel_initializer=ik, bias_initializer=ib, name='dense_'+str(i))(layer)
            layer = ReLU(negative_slope=0.3, name='dense_relu'+str(i))(layer)

        out1 = Dense(3, kernel_initializer=ik, bias_initializer=ib, name='dense_output')(layer)
        self.out = ReLU(negative_slope=1.0, name='dense_output_relu')(out1)

        self._model = Model([self.input, self.target_prediction, self.advantage, self.ratio_f, self.ratio_a], self.out, name=self.name+'_actor_critic')
        self.optimizer = None



        # https://ai.stackexchange.com/questions/18753/how-to-set-the-target-for-the-actor-in-a2c

    def compile(self):
        # LOSS AND OPTIMIZATION

        def actor_critic_loss(output, pred, advantage, ratio_f, ratio_a):
            # Output: network output: force, angle, value
            # Pred: predicted target: used force, use angle, episode return
            # Advantage: advantage: ex.r1 + gamma * pred(v1) - pred(v0)
            # Ratios: importance sampling ratios
            '''
            print("in-loss printouts")
            print(output)
            print(pred)
            print(advantage)
            '''

            # Want critic to predict episode return
            critic_loss = K.pow(pred[0,2] - output[0,2], 2)

            def action_loss(act, pred, rat):
                # Heavy Influence: https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
                '''
                Goal: change probability of action taken in direction of sign(advantage)
                prob = integration_width * prob_density
                density = normal_function = gauss_fac * exp(-0.5*square((x-u)/sig))
                let: x=action taken (model output to controller)
                let: u=predicted action (model output during training)

                Use log probabilities in loss function. We will decrease the loss
                As:
                - in_exponent increases
                - prob [= width * gauss_fac * exp((-)in_exponent)] decreases
                - log(prob) [= log(width) + (-)in_exponent|]   decreases
                - log(prob) [~ (-)in_exponent|] decreases
                - log(prob) [~ (+) in_exponent] increases
                - (-)log(prob) decreases
                - we get closer to where we want to go
                Then:
                - multiple by advantage to control direction we want to go
                - positive advantage: we do want to increase probability, etc
                '''


                delta = act - pred
                expo = 0.5 * K.square(delta/SIG) + 1e-5
                # Expo has a max of ~0
                # Expo has no minimum value
                log_prob = expo # + K.log(GAUSS_FRAC * WIDTH) # These are constant
                log_prob = K.clip(log_prob, 1e-5, 100.0)
                loss = log_prob * advantage
                return loss * rat
            force_loss = action_loss(output[0,0],pred[0,0], ratio_f)
            angle_loss = action_loss(output[0,1],pred[0,1], ratio_a)

            # Not sure why angles tend to go off in one direction yet
            # try to limit for now
            large_angle_loss = K.square(output[0,1])
            neg_speed_loss = -output[0,0]

            return critic_loss + force_loss + angle_loss + large_angle_loss

        self.optimizer = Adam(learning_rate=self.settings.learning.alpha, clipnorm=1.0)
        loss = actor_critic_loss(self.out, self.target_prediction, self.advantage, self.ratio_f, self.ratio_a)
        self._model.add_loss(loss)
        self._model.compile(self.optimizer)
        self._model._make_predict_function()

    def prepare_data_internal(self, data):
        states = np.array([d.state for d in data])
        targets = np.array([d.target for d in data])
        advantages = np.array([d.advantage for d in data])
        ratios_force = np.array([d.ratio_force for d in data])
        ratios_angle = np.array([d.ratio_angle for d in data])
        return (states, targets, advantages, ratios_force, ratios_angle)
    
    def predict(self, data):
        data = self.prepare_data_internal(data)
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            out = self._model.predict(data)
            output = NetworkOutputs()
            output.value = out[0][2]
            output.force = out[0][0]
            output.angle = out[0][1]
            return output
    
    def fit(self, data, verbose, batch_size=1):
        data = self.prepare_data_internal(data)
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            self._model.fit(data, verbose=verbose, batch_size=batch_size)

    def make_dummy_data(self):
        data = NetworkInputs()
        data.state = [0.0]*self.N
        data.target = [0.0,0.0,0.0]
        data.advantage = [0.0]
        data.ratio_force = [1.0]
        data.ratio_angle = [1.0]
        data.ret = [0.0]
        return data

    def dummy_test(self):
        data = self.make_dummy_data()
        data = self.prepare_data_internal([data])
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            self._model.predict(data)

class Network:
    def __init__(self, settings, N):
        self.settings = settings
        self.N = N
        self.util = Utility()
        self.freezing = False

        # MODEL
        self._model = MyModel(self.settings, N, 'model')
        self.frozen_model = MyModel(self.settings, N, 'frozen')

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

        if self.settings.memory.load_saved_network:
            self.load_state()

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
        data = []
        original = []
        gamma = self.settings.learning.gamma

        for ex in exp:
            inputs = NetworkInputs()
            
            s0 = ex.s0
            inputs.state = s0

            pred0 = self.model(s0)
            original.append(pred0)

            '''
            v1 = float(pred1[2])
            if ex.next_terminal:
                advantage = float(ex.r1 - v0)
            else:
                advantage = float(ex.r1 + gamma * v1 - v0)
            '''
            # https://livebook.manning.com/book/deep-learning-and-the-game-of-go/chapter-12/46
            # MC Advantage
            v0 = pred0.value
            advantage = float(ex.G - v0)
            inputs.advantage = [advantage]

            target_critic = ex.G #float(v0 + (ex.r1 + ex.G - v0))
            inputs.target = [ex.a_force, ex.a_angle, target_critic]

            inputs.ret = [ex.G]

            bf = clip(ex.pf, 1e-3, 1.0-1e-3)
            pf = self.util.normal_int_prob(ex.a_force, float(pred0.force), SIG)
            pf = clip(pf, 1e-3, 1.0-1e-3)
            rat_f = clip(pf / bf, 0.1, 2.0)
            inputs.ratio_force = [rat_f]

            ba = clip(ex.pa, 1e-3, 1.0-1e-3)
            pa = self.util.normal_int_prob(ex.a_angle, float(pred0.angle), SIG)
            pa = clip(pa, 1e-3, 1.0-1e-3)
            rat_a = clip(pa / ba, 0.1, 2.0)
            inputs.ratio_angle = [rat_a]

            data.append(inputs)

        return data, original


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

        new = [self.model(dat) for dat in data]

        for orig, dat, newP in zip(original[0:5], data[0:5], new[0:5]):
            ret = dat.ret
            target = dat.target
            adv = dat.advantage
            print("orig(%f,%f,%f)->ret(%f)->target(%f,%f,%f)->adv(%f)->(%f,%f,%f)" % (orig[0],orig[1],orig[2], ret, target[0][0], target[0][1], target[0][2], adv, newP[0][0], newP[0][1], newP[0][2]))

        sample_results= []
        for orig, newP in zip(original, new):
            results = SampleTrainingResults()
            results.c_step = float(newP[0][2] - orig[2])
            results.af_step = float(newP[0][0] - orig[0])
            results.aa_step = float(newP[0][1] - orig[1])
            sample_results.append(results)

            bad = False
            bad = bad or  math.isnan(float(newP[0][0]))
            bad = bad or  math.isnan(float(newP[0][1]))
            bad = bad or  math.isnan(float(newP[0][2]))

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
        memory_dir = self.settings._files.root_dir + "/memory"
        
        if self.settings.memory.load_saved_network:
            try:
                network_file = memory_dir + "/model.h5"
                self.graph = tf.compat.v1.Graph()
                self._model.load_model(network_file)
                self.freeze()
                print("Loaded Network")
                return True
            except OSError as e:
                print("Could not load network from file %s: %s" % (network_file, str(e)))
                return False
        return False

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
        
