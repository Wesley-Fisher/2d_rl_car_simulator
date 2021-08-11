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
