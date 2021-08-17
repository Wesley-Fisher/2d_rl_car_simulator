
from rl_car_simulator.settings import CONSTANTS
import numpy as np

from .utilities import Utility

class CarState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0

        self.h = 0.0
        self.vh = 0.0

class ControlAction:
    def __init__(self):
        pass
    def get_applied_action_ext(self):
        raise NotImplementedError
    def get_action_int(self):
        raise NotImplementedError
    def get_prob(self):
        raise NotImplementedError
    def get_prob_of_int_action(self, action):
        raise NotImplementedError
    def get_random_elements(self):
        raise NotImplementedError
    def apply_noise(self, noise):
        raise NotImplementedError

class DirectControlAction:
    def __init__(self, a, p):
        self.action = a
        self.prob = p
    def get_applied_action_ext(self):
        return self.action
    def get_action_int(self):
        return self.action
    def get_prob(self):
        return self.prob
    def get_prob_of_int_action(self, action):
        return Utility().normal_int_prob(action, self.action, CONSTANTS.sigma)
    def get_random_elements(self):
        return 1
    def apply_noise(self, noise):
        act_orig = self.action
        self.action = act_orig + noise[0]
        self.prob = Utility().normal_int_prob(act_orig, self.action, CONSTANTS.sigma)


class CarControls:
    def __init__(self, force, angle):
        self.force = force
        self.angle = angle

class Car:
    def __init__(self, settings, state):
        self.settings = settings
        self.state = state
        self.controls = CarControls(DirectControlAction(0.0, 1.0), DirectControlAction(0.0, 1.0))
        self.controller = None

        self.util = Utility()
        self.episode_steps = []
        self.step_experience = None
        self.goal = (0,0)
        self.goal_id = -1

        self.sensed_state = None
        self.lidar_state = None

        self.collided = False
        self.reached_goal = False
        self.too_old = False

        self.type =""
        self.name = ""

    def set_controller(self, controller):
        self.controller = controller

    def get_center(self):
        c = np.array([[self.state.x],[self.state.y]])
        return c

    def set_goal(self, goal, goal_id):
        self.goal = goal
        self.goal_id = goal_id

    def get_corners(self):
        dl = self.settings.car_properties.length * 0.5
        dw = self.settings.car_properties.width * 0.5

        R = self.util.rot(self.state.h)
        c = self.get_center()

        corners = []
        for lr, fb in [(1.0, 1.0),(1.0, -1.0),(-1.0, -1.0),(-1.0, 1.0)]:
            delta = np.array([[fb*dl], [lr*dw]])
            dir = np.dot(R, delta)
            corner = c + dir
            corner = corner.reshape(1,2)
            corners.append(corner)
        return corners

    def get_windshield_corners(self):
        dl = self.settings.car_properties.length * 0.5
        dw = self.settings.car_properties.width * 0.5

        R = self.util.rot(self.state.h)
        c = self.get_center()
        side = 0.85
        front = 0.85
        back = 0.3
        corners = []
        for lr, fb in [(side, front),(side, back),(-side, back),(-side, front)]:
            delta = np.array([[fb*dl], [lr*dw]])
            corner = c + np.dot(R, delta)
            corners.append(corner)
        return corners

    def set_controls(self, controls):
        self.controls = controls

    def get_controls(self, state):
        return self.controller.get_car_control(state)

    def set_type(self, type):
        self.type = type

    def set_name(self, name):
        self.name = name
    
    def get_type(self):
        return self.type

    def get_name(self):
        return self.name


class CarStepExperience:
    def __init__(self):
        self.s0 = None
        self.a0 = None
        self.r1 = None
        self.s1 = None
        self.pf = None
        self.pa = None
    
    def set_s0(self, s):
        self.s0 = s
    
    def set_a0(self, a):
        self.a0 = a
    
    def set_r1(self, r):
        self.r1 = r
    
    def set_s1(self, s):
        self.s1 = s