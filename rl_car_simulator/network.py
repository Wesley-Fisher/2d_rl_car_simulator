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
        self._model = Model([self.input, self.target_prediction, self.advantage, self.ratios_f, self.ratios_a], self.out, name=self.name+'_actor_critic')
        if compile:
            self.compile()

    def save_model(self, filename):
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            self._model.save(filename)

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
        self.ratios_f = Input(shape=(1), name=self.name+'_rat_f_in')
        self.ratios_a = Input(shape=(1), name=self.name+'_rat_a_in')

        layer = Dense(WN, kernel_initializer=ik, bias_initializer=ib)(self.input)
        layer = ReLU(negative_slope=0.3)(layer)

        for i in range(0, self.D):
            layer = Dense(WN, kernel_initializer=ik, bias_initializer=ib)(layer)
            layer = ReLU(negative_slope=0.3)(layer)

        out1 = Dense(3, kernel_initializer=ik, bias_initializer=ib)(layer)
        self.out = ReLU(negative_slope=1.0)(out1)

        self._model = Model([self.input, self.target_prediction, self.advantage, self.ratios_f, self.ratios_a], self.out, name=self.name+'_actor_critic')
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

            return critic_loss + force_loss + angle_loss

        self.optimizer = Adam(learning_rate=self.settings.learning.alpha, clipnorm=1.0)
        loss = actor_critic_loss(self.out, self.target_prediction, self.advantage, self.ratios_f, self.ratios_a)
        self._model.add_loss(loss)
        self._model.compile(self.optimizer)
        self._model._make_predict_function()
    
    def predict(self, data):
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            return self._model.predict(data)
    
    def fit(self, data, verbose, batch_size=1):
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            self._model.fit(data, verbose=verbose, batch_size=batch_size)

    def dummy_test(self):
        state = np.array([ [0.0]*self.N])
        target = np.array([[0.0,0.0,0.0]])
        adv = np.array([[0.0]])
        rat_f = np.array([[1.0]])
        rat_a = np.array([[1.0]])
        data = (state, target, adv, rat_f, rat_a)
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
        


    def model(self, state, model=None):
        dummy_target = np.array([[1,1,1]])
        dummy_advantage = np.array([[0]])
        dummy_rat_f = np.array([[1.0]])
        dummy_rat_a = np.array([[1.0]])
        data = (state, dummy_target, dummy_advantage, dummy_rat_f, dummy_rat_a)

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
            tens = self._model.predict(data)
        else:
            tens = model.predict(data)
        a0 = float(tens[0][0])
        a1 = float(tens[0][1])
        v = float(tens[0][2])
        return [np.array([[a0], [a1], [v]])]

    def predict_advantage(self, ex):
        _, _, _, advantages, _, _, _ = self.build_epoch_targets([ex])
        return advantages[0]

    def freeze(self):
        self.freezing = True
        self.frozen_model.copy_weights(self._model.get_weights())
        self.freezing = False
    
    def get(self, x):
        while self.freezing:
            time.sleep(0.001)
        x = x.reshape((1,self.N))
        return self.model(x, self.frozen_model)
    
    def add_experience(self, exp):
        self.new_training_experiences = self.new_training_experiences + exp

    def build_epoch_targets(self, exp):
        states = []
        original = []
        targets = []
        advantages = []
        returns = []
        ratios_force = []
        ratios_angle = []

        gamma = self.settings.learning.gamma

        for ex in exp:
            s0 = ex.s0
            states.append(s0)

            pred0 = self.model(s0)[0]
            pred1 = self.model(ex.s1)[0]
            original.append(pred0)

            v0 = float(pred0[2])
            v1 = float(pred1[2])

            target_critic = ex.G#float(v0 + (ex.r1 + ex.G - v0))

            '''
            if ex.next_terminal:
                advantage = float(ex.r1 - v0)
            else:
                advantage = float(ex.r1 + gamma * v1 - v0)
            '''
            # 
            advantage = float(ex.G - v0) #https://livebook.manning.com/book/deep-learning-and-the-game-of-go/chapter-12/46
            advantages.append(advantage)

            target = np.array([[ex.a_force, ex.a_angle, target_critic]])
            targets.append(target)

            returns.append(ex.G)

            bf = ex.pf
            pf = self.util.normal_int_prob(ex.a_force, float(pred0[0]), SIG)
            rat_f = clip(pf / bf, 0.1, 2.0)
            ratios_force.append(rat_f)

            ba = ex.pa
            pa = self.util.normal_int_prob(ex.a_angle, float(pred0[1]), SIG)
            rat_a = clip(pa / ba, 0.1, 2.0)
            ratios_angle.append(rat_a)


        return states, original, targets, advantages, returns, ratios_force, ratios_angle


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

    def fit_model(self, states, targets, advantages, ratios_f, ratios_a, verbose=0):

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
        ratios_f = numpify(ratios_f, 1)
        ratios_a = numpify(ratios_a, 1)
        data = (states, targets, advantages, ratios_f, ratios_a)
        
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

        states, original, targets, advantages, returns, ratios_f, ratios_a = self.build_epoch_targets(self.training_experience)

        self.fit_model(states, targets, advantages, ratios_f, ratios_a)

        new = [self.model(state) for state in states]

        for orig, target, adv, newP, ret in zip(original[0:5], targets[0:5], advantages[0:5], new[0:5], returns[0:5]):
            print("orig(%f,%f,%f)->ret(%f)->target(%f,%f,%f)->adv(%f)->(%f,%f,%f)" % (orig[0],orig[1],orig[2], ret, target[0][0], target[0][1], target[0][2], adv, newP[0][0], newP[0][1], newP[0][2]))

        sample_results= []
        for orig, new in zip(original, new):
            results = SampleTrainingResults()
            results.c_step = float(new[0][2] - orig[2])
            results.af_step = float(new[0][0] - orig[0])
            results.aa_step = float(new[0][1] - orig[1])
            sample_results.append(results)

            bad = False
            bad = bad or  math.isnan(float(new[0][0]))
            bad = bad or  math.isnan(float(new[0][1]))
            bad = bad or  math.isnan(float(new[0][2]))

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
        self._model.save_model(network_file)

    def load_state(self):
        if self.settings._files.root_dir is None:
            return
        memory_dir = self.settings._files.root_dir + "/memory"

        
        if self.settings.memory.load_saved_network:
            try:
                network_file = memory_dir + "/model.h5"
                self.graph = tf.compat.v1.Graph()
                self._model.load_model(network_file)
                self.freeze()
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


        
