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
from tensorflow.keras.layers import Dense, ReLU, Input, Softmax
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.ops.gen_math_ops import is_nan 
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import set_session

from .utilities import Utility
from .car import ControlAction



SIG = CONSTANTS.sigma
WIDTH = Utility().normal_int_width(SIG)
GAUSS_FRAC = 1.0 / (SIG * math.sqrt(2*math.pi))

class NetworkInputs:
    def __init__(self):
        raise NotImplementedError
        self.state = None
        self.target = None
        self.advantage = None
        self.ratio_force = None
        self.ratio_angle = None
        self.ret = None

class NetworkAction(ControlAction):
    def __init__(self):
        self.action = 0.0
        self.prob = 0.0
    def get_random_elements(self):
        raise NotImplementedError
    def get_applied_action_ext(self):
        raise NotImplementedError
    def get_action_int(self):
        raise NotImplementedError
    def get_prob(self):
        raise NotImplementedError
    def apply_noise(self, noise):
        raise NotImplementedError
    def get_prob_of_int_action(self, action):
        raise NotImplementedError


class SoftmaxNetworkAction(NetworkAction):
    def __init__(self, scale):
        self.action = 0
        self.prob = 0.0
        self.scale = scale
        self.map = {0: -1, 1: 0, 2: 1}
    def get_random_elements(self):
        return 3
    def get_applied_action_ext(self):
        i = self.action.index(max(self.action))
        pol = self.map[i]
        return pol * self.scale
    def get_action_int(self):
        return self.action
    def get_prob(self):
        i = self.action.index(max(self.action))
        return self.action[i]
    def apply_noise(self, noise):
        tot = 0.0
        for i in range(0, len(noise)):
            self.action[i] = self.action[i] + noise[0]
            tot = tot + self.action[i]*self.action[i]
        tot = math.sqrt(tot)
        for i in range(0, len(noise)):
            self.action[i] = self.action[i] / tot
    def get_prob_of_int_action(self, action):
        i = self.action.index(max(self.action))
        return self.action[i]


class NetworkOutputs:
    def __init__(self):
        self.value = 0.0
        self.force = NetworkAction()
        self.action = NetworkAction()

class MyModel:
    def __init__(self, settings,N, name):
        raise NotImplementedError

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
        raise NotImplementedError

    def save_model(self, filename):
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            self._model.save(filename, save_format='h5')

    def make_model(self):
        raise NotImplementedError

    def compile(self):
        raise NotImplementedError

    def prepare_data_internal(self, data):
        raise NotImplementedError
    
    def predict(self, data):
        raise NotImplementedError
    
    def fit(self, data, verbose, batch_size=1):
        data = self.prepare_data_internal(data)
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            self._model.fit(data, verbose=verbose, batch_size=batch_size)

    def make_dummy_data(self):
        NotImplementedError

    def dummy_test(self):
        data = self.make_dummy_data()
        data = self.prepare_data_internal([data])
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            self._model.predict(data)

    def build_epoch_targets(self, exp):
        raise NotImplementedError

    def get_force_action_prob(self, action):
        raise NotImplementedError

    def get_angle_action_prob(self, action):
        raise NotImplementedError
