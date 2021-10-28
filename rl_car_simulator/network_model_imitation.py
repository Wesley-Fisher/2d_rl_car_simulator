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
from .car import ControlAction, DiscreteControlAction
from .network_model import MyModel, NetworkAction, NetworkInputs, NetworkOutputs



SIG = CONSTANTS.sigma
WIDTH = Utility().normal_int_width(SIG)
GAUSS_FRAC = 1.0 / (SIG * math.sqrt(2*math.pi))

class ImitationNetworkInputs(NetworkInputs):
    def __init__(self):
        self.state = None
        self.force = None
        self.angle = None

        self.ret = [0.0]
        self.advantage = [0.0]

class ImitationNetworkAction(DiscreteControlAction):
    def __init__(self, scale):
        self.action = [0.23, 0.5, 0.25]
        self.scale = scale
        self.ind_to_action = {0: -self.scale, 1: 0.0, 2: self.scale }
    def get_random_elements(self):
        return 3
    def get_applied_action_ext(self):
        i = self.get_action_index()
        return self.ind_to_action[i]
    def get_action_int(self):
        ret = [0.0, 0.0, 0.0]
        i = self.get_action_index()
        ret[i] = 1.0
        return ret
    def get_prob(self):
        i = self.get_action_index()
        return self.action[i]
    def apply_noise(self, noise):
        tot = 0.0
        for i in range(0, len(noise)):
            self.action[i] = self.action[i] + noise[i]
            tot = tot + self.action[i]
        for i in range(0, len(self.action)):
            self.action[i] = self.action[i] / tot
    def get_prob_of_int_action(self, action):
        i = action.index(max(action))
        return self.action[i]
    def get_action_index(self):
        return self.action.index(max(self.action))


class MyImitationModel(MyModel):
    def __init__(self, settings,N, name):
        self.settings = settings
        self.N = N
        self.name = name
        self.util = Utility()

        self.W = settings.network.W
        self.D = settings.network.D

        self.graph = tf.Graph()
        self.session = tf.compat.v1.Session(graph=self.graph)

        self.state_input = None
        self.applied_force_input = None
        self.applied_angle_input = None

        self.force_out = None
        self.angle_out = None

        self._model = None
        self.optimizer=None

        with self.graph.as_default():
            set_session(self.session)
            self.make_model()
            self.compile()

    def handle_new_model(self, compile=True):
        inputs = self._model.inputs
        outputs = self._model.outputs

        self.state_input = inputs[0]
        self.applied_force_input = inputs[1]
        self.applied_angle_input = inputs[2]

        self.force_out = outputs[0]
        self.angle_out = outputs[1]

        inputs = [self.state_input,
                  self.applied_force_input,
                  self.applied_angle_input]
        outputs = [self.force_out,
                   self.angle_out]
        self._model = Model(inputs, outputs, name=self.name+'_imitation')
        if compile:
            self.compile()


    def make_model(self):
        # MODEL NETWORK
        ik = initializers.RandomNormal(stddev=0.1, seed=1)
        ib = initializers.RandomNormal(stddev=0.1, seed=2)
        WN = int(self.W*self.N + 1)

        self.state_input = Input(shape=(self.N), name=self.name+'_state_in')
        self.applied_force_input = Input(shape=(3), name=self.name+'_force_in')
        self.applied_angle_input = Input(shape=(3), name=self.name+'_angle_in')

        inputs = [self.state_input,
                  self.applied_force_input,
                  self.applied_angle_input]

        layer = Dense(WN, kernel_initializer=ik, bias_initializer=ib, name="dense_input")(self.state_input)
        layer = ReLU(negative_slope=0.3, name="dense_input_relu")(layer)

        for i in range(0, self.D):
            layer = Dense(WN, kernel_initializer=ik, bias_initializer=ib, name='dense_'+str(i))(layer)
            layer = ReLU(negative_slope=0.3, name='dense_relu'+str(i))(layer)

        out1 = Dense(3, kernel_initializer=ik, bias_initializer=ib, name='dense_output')(layer)

        force_out = Dense(3, kernel_initializer=ik, bias_initializer=ib, name='force_dense_out')(out1)
        self.force_out = Softmax(name='force_output')(force_out)

        angle_out = Dense(3, kernel_initializer=ik, bias_initializer=ib, name='angle_dense_out')(out1)
        self.angle_out = Softmax(name='angle_output')(angle_out)

        outputs = [self.force_out, self.angle_out]

        self._model = Model(inputs, outputs, name=self.name+'_imitation')
        self.optimizer = None


    def compile(self):
        # LOSS AND OPTIMIZATION

        def actor_critic_loss(force_out, angle_out, force_used, angle_used):
            def action_loss(act, pred):
                
                # Get full difference
                diff = act - pred

                # Get difference only of applied action
                #diff = act - pred*act

                return K.dot(diff, K.transpose(diff))

            force_loss = action_loss(force_used, force_out)
            angle_loss = action_loss(angle_used, angle_out)

            return force_loss + angle_loss

        self.optimizer = Adam(learning_rate=self.settings.learning.alpha, clipnorm=1.0)
        loss = actor_critic_loss(self.force_out,
                                 self.angle_out,
                                 self.applied_force_input,
                                 self.applied_angle_input)
        self._model.add_loss(loss)
        self._model.compile(self.optimizer)
        self._model._make_predict_function()

    def prepare_data_internal(self, data):
        states = np.array([np.array(d.state) for d in data])
        forces = np.array([np.array(d.force) for d in data])
        angles = np.array([np.array(d.angle)for d in data])
        return (states, forces, angles)
    
    def predict(self, data):
        data = self.prepare_data_internal(data)
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            out = self._model.predict(data)
            output = NetworkOutputs()
            output.value = 0.0

            force = ImitationNetworkAction(self.settings.keyboard.force)
            force.action = out[0][0].tolist()
            output.force = force

            angle = ImitationNetworkAction(self.settings.keyboard.angle)
            angle.action = out[1][0].tolist()
            output.angle = angle
            return output
    

    def make_dummy_data(self):
        data = ImitationNetworkInputs()
        data.state = [0.0]*self.N
        data.force = [0.0, 0.0, 0.0]
        data.angle = [0.0, 0.0, 0.0]
        return data


    def build_epoch_targets(self, exp):
        data = []
        original = []
        gamma = self.settings.learning.gamma

        for ex in exp:
            inputs = ImitationNetworkInputs()
            
            s0 = ex.s0
            inputs.state = s0
            inputs.force = ex.action_force.get_action_int()
            inputs.angle = ex.action_angle.get_action_int()

            pred_data = self.make_dummy_data()
            pred_data.state = s0
            pred0 = self.predict([pred_data])
            original.append(pred0) 

            data.append(inputs)

        return data, original

    def get_force_action_prob(self, action):
        return action.value, action.prob

    def get_angle_action_prob(self, action):
        return action.value, action.prob