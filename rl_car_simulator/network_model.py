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
from .car import ControlAction



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

class NetworkAction(ControlAction):
    def __init__(self):
        self.action = 0.0
        self.prob = 0.0
    def get_random_elements(self):
        return 1
    def get_applied_action_ext(self):
        return self.action
    def get_action_int(self):
        return self.action
    def get_prob(self):
        return self.prob
    def apply_noise(self, noise):
        act_orig = self.action
        self.action = act_orig + noise[0]
        self.prob = Utility().normal_int_prob(act_orig, self.action, CONSTANTS.sigma)
    def get_prob_of_int_action(self, action):
        return Utility().normal_int_prob(action, self.action, CONSTANTS.sigma)


class NetworkOutputs:
    def __init__(self):
        self.value = 0.0
        self.force = NetworkAction()
        self.action = NetworkAction()

class MyModel:
    def __init__(self, settings,N, name):
        self.settings = settings
        self.N = N
        self.name = name
        self.util = Utility()

        self.graph = tf.Graph()
        self.session = tf.compat.v1.Session(graph=self.graph)

        self.state_input = None
        self.applied_force_input = None
        self.applied_angle_input = None
        self.critic_target_input = None
        self.advantage_input = None
        self.ratio_f_input = None
        self.ratio_a_input = None

        self.force_out = None
        self.angle_out = None
        self.value_prediction = None

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
        outputs = self._model.outputs

        self.state_input = inputs[0]
        self.applied_force_input = inputs[1]
        self.applied_angle_input = inputs[2]
        self.critic_target_input = inputs[3]
        self.advantage_input = inputs[4]
        self.ratio_f_input = inputs[5]
        self.ratio_a_input = inputs[6]

        self.force_out = outputs[0]
        self.angle_out = outputs[1]
        self.value_prediction = outputs[2]
        '''
        print(self.input)
        print(self.target_prediction)
        print(self.advantage)
        '''
        inputs = [self.state_input,
                  self.applied_force_input,
                  self.applied_angle_input,
                  self.critic_target_input,
                  self.advantage_input,
                  self.ratio_f_input,
                  self.ratio_a_input]
        outputs = [self.force_out,
                   self.angle_out,
                   self.value_prediction]
        self._model = Model(inputs, outputs, name=self.name+'_actor_critic')
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

        self.state_input = Input(shape=(self.N), name=self.name+'_state_in')
        self.applied_force_input = Input(shape=(1), name=self.name+'_force_in')
        self.applied_angle_input = Input(shape=(1), name=self.name+'_angle_in')
        self.critic_target_input = Input(shape=(1), name=self.name+'_critic_in')
        self.advantage_input = Input(shape=(1), name=self.name+'_advantage_in')
        self.ratio_f_input = Input(shape=(1), name=self.name+'_ratio_f_in')
        self.ratio_a_input = Input(shape=(1), name=self.name+'_ratio_a_in')

        inputs = [self.state_input,
                  self.applied_force_input,
                  self.applied_angle_input,
                  self.critic_target_input,
                  self.advantage_input,
                  self.ratio_f_input,
                  self.ratio_a_input]

        layer = Dense(WN, kernel_initializer=ik, bias_initializer=ib, name="dense_input")(self.state_input)
        layer = ReLU(negative_slope=0.3, name="dense_input_relu")(layer)

        for i in range(0, self.D):
            layer = Dense(WN, kernel_initializer=ik, bias_initializer=ib, name='dense_'+str(i))(layer)
            layer = ReLU(negative_slope=0.3, name='dense_relu'+str(i))(layer)

        out1 = Dense(3, kernel_initializer=ik, bias_initializer=ib, name='dense_output')(layer)

        force_out = Dense(1, kernel_initializer=ik, bias_initializer=ib, name='force_dense_out')(out1)
        self.force_out = ReLU(negative_slope=1.0, name='force_output')(force_out)

        angle_out = Dense(1, kernel_initializer=ik, bias_initializer=ib, name='angle_dense_out')(out1)
        self.angle_out = ReLU(negative_slope=1.0, name='angle_output')(angle_out)

        value_prediction = Dense(1, kernel_initializer=ik, bias_initializer=ib, name='value_dense_out')(out1)
        self.value_prediction = ReLU(negative_slope=1.0, name='value_output')(value_prediction)

        outputs = [self.force_out, self.angle_out, self.value_prediction]

        self._model = Model(inputs, outputs, name=self.name+'_actor_critic')
        self.optimizer = None



        # https://ai.stackexchange.com/questions/18753/how-to-set-the-target-for-the-actor-in-a2c

    def compile(self):
        # LOSS AND OPTIMIZATION

        def actor_critic_loss(force_out, angle_out, value_out, force_used, angle_used, return_value, advantage, ratio_f, ratio_a):
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
            critic_loss = K.pow(return_value - value_out, 2)

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
            force_loss = action_loss(force_out, force_used, ratio_f)
            angle_loss = action_loss(angle_out, angle_used, ratio_a)

            # Not sure why angles tend to go off in one direction yet
            # try to limit for now
            large_angle_loss = K.square(angle_out)
            neg_speed_loss = -force_out

            return critic_loss + force_loss + angle_loss + large_angle_loss

        self.optimizer = Adam(learning_rate=self.settings.learning.alpha, clipnorm=1.0)
        loss = actor_critic_loss(self.force_out,
                                 self.angle_out,
                                 self.value_prediction,
                                 self.applied_force_input,
                                 self.applied_angle_input,
                                 self.critic_target_input,
                                 self.advantage_input,
                                 self.ratio_f_input,
                                 self.ratio_a_input)
        self._model.add_loss(loss)
        self._model.compile(self.optimizer)
        self._model._make_predict_function()

    def prepare_data_internal(self, data):
        states = np.array([d.state for d in data])
        forces = np.array([d.target[0] for d in data])
        angles = np.array([d.target[1] for d in data])
        values = np.array([d.target[2] for d in data])
        advantages = np.array([d.advantage[0] for d in data])
        ratios_force = np.array([d.ratio_force for d in data])
        ratios_angle = np.array([d.ratio_angle for d in data])
        return (states, forces, angles, values, advantages, ratios_force, ratios_angle)
    
    def predict(self, data):
        data = self.prepare_data_internal(data)
        with self.graph.as_default(), self.session.as_default():
            set_session(self.session)
            out = self._model.predict(data)
            output = NetworkOutputs()
            output.value = out[2][0]

            force = NetworkAction()
            force.action = out[0][0]
            force.action = self.util.normal_int_prob(out[0][0], out[0][0], self.settings.statistics.sigma)
            output.force = force

            angle = NetworkAction()
            angle.action = out[0][0]
            angle.action = self.util.normal_int_prob(out[1][0], out[1][0], self.settings.statistics.sigma)
            output.angle = angle
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

    def build_epoch_targets(self, exp):
        data = []
        original = []
        gamma = self.settings.learning.gamma

        for ex in exp:
            inputs = NetworkInputs()
            
            s0 = ex.s0
            inputs.state = s0

            pred_data = self.make_dummy_data()
            pred_data.state = s0
            pred0 = self.predict([pred_data])
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
            inputs.target = [ex.action_force.get_action_int(),
                             ex.action_angle.get_action_int(),
                             target_critic]

            inputs.ret = [ex.G]

            EPS = 1e-3

            bf = clip(ex.action_force.get_prob(), EPS, 1.0-EPS)
            pf = ex.action_force.get_prob_of_int_action(pred0.force.get_action_int())
            pf = clip(pf, EPS, 1.0-EPS)
            rat_f = clip(pf / bf, 0.1, 2.0)
            inputs.ratio_force = [rat_f]

            ba = clip(ex.action_angle.get_prob(), EPS, 1.0-EPS)
            pa = ex.action_angle.get_prob_of_int_action(pred0.angle.get_action_int())
            pa = clip(pf, EPS, 1.0-EPS)
            rat_a = clip(pa / ba, 0.1, 2.0)
            inputs.ratio_angle = [rat_a]

            data.append(inputs)

        return data, original

    def get_force_action_prob(self, action):
        return action.value, action.prob

    def get_angle_action_prob(self, action):
        return action.value, action.prob
