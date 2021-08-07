import keyboard
import random
import math

from .car import CarControls
from .utilities import Utility

class ControllerTypes:
    keyboard = "Keyboard"
    network = "Network"
    hardcoded = "HardCoded"
    random = "Random"
    feedback = "Feedback"
    network_exploration = "NetworkExploration"
    feedback_exploration = "FeedbackExploration"

class Controller:
    def __init__(self, settings):
        self.settings = settings
        self.util = Utility()
        self.stat_p = self.util.normal_density(0, 0, self.settings.statistics.sigma)
        self.stat_p = self.stat_p * self.util.normal_int_width(self.settings.statistics.sigma)

    def get_controls(self, state):
        raise NotImplementedError

    def get_car_control(self, state):
        control = self.get_controls(state)
        control.force = min(max(control.force, -self.settings.keyboard.force), self.settings.keyboard.force)
        control.steer = min(max(control.steer, -self.settings.keyboard.angle), self.settings.keyboard.angle)
        return control

class KeyboardController(Controller):
    def __init__(self, settings):
        Controller.__init__(self, settings)
        self.settings = settings


    def get_controls(self, state):
        w = keyboard.is_pressed('w')
        s = keyboard.is_pressed('s')
        force = float(w - s) * self.settings.keyboard.force
        
        l = keyboard.is_pressed('a')
        r = keyboard.is_pressed('d')
        angle = float(r - l) * self.settings.keyboard.angle

        return CarControls(force, angle, self.stat_p, self.stat_p)

class NetworkController(Controller):
    def __init__(self, settings, network):
        Controller.__init__(self, settings)
        self.settings = settings
        self.network = network

    def get_controls(self, state):
        control = self.network.get(state)
        #print(control)
        force = float(control[0][0]) + random.gauss(0.0, 0.1*self.settings.statistics.sigma)
        angle = float(control[0][1]) + random.gauss(0.0, 0.1*self.settings.statistics.sigma)
        force = min(max(force, -self.settings.keyboard.force), self.settings.keyboard.force)
        angle = min(max(angle, -self.settings.keyboard.angle), self.settings.keyboard.angle)
        if math.isnan(force):
            force = 0.0
        if math.isnan(angle):
            angle = 0.0

        # Network follows exactly what it predicts
        return CarControls(force, angle, self.stat_p, self.stat_p)

class HardCodedController(Controller):
    def __init__(self, settings, f, a):
        Controller.__init__(self, settings)
        self.settings = settings
        self.f = f
        self.a = a

    def get_controls(self, state):
        # Hardcoded has 100% probability of taking this action
        return CarControls(self.f, self.a, 1.0, 1.0)

class RandomController(Controller):
    def __init__(self, settings, force_bias_range=2, force_step=0.5, angle_bias_range=0.5, angle_step=0.1):
        Controller.__init__(self, settings)
        self.settings = settings
        self.a = 0.0
        self.f = 0.0
        self.force_bias_range = force_bias_range
        self.force_step = force_step
        self.angle_bias_range = angle_bias_range
        self.angle_step = angle_step
        self.reset()

    def reset(self):
        self.a = random.uniform(-self.angle_bias_range, self.angle_bias_range)
        self.f = random.uniform(-self.force_bias_range, self.force_bias_range)
    
    def get_controls(self, state):
        self.a = self.a + random.gauss(0, self.angle_step * self.settings.physics.control_timestep)
        self.f = self.f + random.gauss(0, self.force_step * self.settings.physics.control_timestep)
        # Stick to default probability for now
        return CarControls(self.f, self.a, self.stat_p, self.stat_p)

class FeedbackController(Controller):
    def __init__(self, settings):
        Controller.__init__(self, settings)
        self.settings = settings
    
    def get_controls(self, state):
        dist = state[3]
        d_head = state[4] - state[2]

        if d_head > math.pi:
            d_head = -2 * math.pi + d_head
        if d_head < -math.pi:
            d_head = 2 * math.pi + d_head

        if abs(d_head) > 2 * dist:
            force = -0.5
            angle = 0.0
            return CarControls(force, angle, 1.0, 1.0)

        force = self.settings.feedback_car.force
        angle = self.settings.feedback_car.k * d_head

        def close(lidars):
            for i in lidars:
                d = state[5 + i] # Base state + index of lidar
                if d < self.settings.feedback_car.close:
                    return True
        
        left = close(self.settings.feedback_car.left_lidars)
        front = close(self.settings.feedback_car.front_lidars)
        right = close(self.settings.feedback_car.right_lidars)

        if left:
            angle = 0.5
        if right:
            angle = -0.5
        if front:
            if left:
                angle = 0.5
            else:
                angle = -0.5
        
        if left and front and right:
            dl = state[5 + self.settings.feedback_car.left_lidars[0]]
            dr = state[5 + self.settings.feedback_car.right_lidars[0]]
            angle = dr - dl
            force = -2

        return CarControls(force, angle, 1.0, 1.0)


class ExplorationController(Controller):
    def __init__(self, settings, base):
        Controller.__init__(self, settings)
        self.settings = settings
        self.base = base
        self.rnd = RandomController(settings, self.settings.exploration.force_bias_range,
                                              self.settings.exploration.force_step,
                                              self.settings.exploration.angle_bias_range,
                                              self.settings.exploration.angle_step)

    def reset(self):
        self.rnd.reset

    def get_car_control(self, state):
        c1 = self.base.get_car_control(state)
        c2 = self.rnd.get_car_control(state)

        f = c1.force + c2.force
        a = c1.steer + c2.steer

        pf = self.util.normal_int_prob(f, c1.force, self.settings.statistics.sigma)
        pa = self.util.normal_int_prob(a, c1.steer, self.settings.statistics.sigma)
        return CarControls(f, a, pf, pa)


class Controllers:
    def __init__(self, keyboard, network, hardcoded, random, feedback, network_exploration, feedback_exploration):
        self.keyboard = keyboard
        self.network = network
        self.hardcoded = hardcoded
        self.random = random
        self.feedback = feedback
        self.network_exploration = network_exploration
        self.feedback_exploration = feedback_exploration

