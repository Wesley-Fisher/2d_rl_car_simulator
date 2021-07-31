import keyboard
import random
import math

from .car import CarControls

class ControllerTypes:
    keyboard = "Keyboard"
    network = "Network"
    hardcoded = "HardCoded"
    random = "Random"
    feedback = "Feedback"

class Controller:
    def __init__(self, settings):
        self.settings = settings

    def get_controls(self, state):
        raise NotImplementedError

    def get_car_control(self, state):
        control = self.get_controls(state)
        control.force = min(max(control.force, -self.settings.keyboard.force), self.settings.keyboard.force)
        control.steer = min(max(control.steer, -self.settings.keyboard.angle), self.settings.keyboard.angle)
        return control

class KeyboardController(Controller):
    def __init__(self, settings):
        self.settings = settings

    def get_controls(self, state):
        w = keyboard.is_pressed('w')
        s = keyboard.is_pressed('s')
        force = float(w - s) * self.settings.keyboard.force
        
        l = keyboard.is_pressed('a')
        r = keyboard.is_pressed('d')
        angle = float(r - l) * self.settings.keyboard.angle
        return CarControls(force, angle)

class NetworkController(Controller):
    def __init__(self, settings, network):
        self.settings = settings
        self.network = network

    def get_controls(self, state):
        control = self.network.get(state)
        #print(control)
        force = control[0][0] + random.gauss(0.0, 0.1*self.settings.statistics.sigma)
        angle = control[0][1] + random.gauss(0.0, 0.1*self.settings.statistics.sigma)
        force = min(max(force, -self.settings.keyboard.force), self.settings.keyboard.force)
        angle = min(max(angle, -self.settings.keyboard.angle), self.settings.keyboard.angle)
        if math.isnan(force):
            force = 0.0
        if math.isnan(angle):
            angle = 0.0
        return CarControls(force, angle)

class HardCodedController(Controller):
    def __init__(self, settings, f, a):
        self.settings = settings
        self.f = f
        self.a = a

    def get_controls(self, state):
        return CarControls(self.f, self.a)

class RandomController(Controller):
    def __init__(self, settings):
        self.settings = settings
        self.a = 0.0
        self.f = 0.0
        self.reset()

    def reset(self):
        self.a = random.uniform(-2, 2)
        self.f = random.uniform(-2, 2)
    
    def get_controls(self, state):
        self.a = self.a + random.gauss(0, 0.5 * self.settings.physics.control_timestep)
        self.f = self.f + random.gauss(0, 0.5 * self.settings.physics.control_timestep)
        return CarControls(self.f, self.a)

class FeedbackController(Controller):
    def __init__(self, settings):
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
            return CarControls(force, angle)

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

        return CarControls(force, angle)


class Controllers:
    def __init__(self, keyboard, network, hardcoded, random, feedback):
        self.keyboard = keyboard
        self.network = network
        self.hardcoded = hardcoded
        self.random = random
        self.feedback = feedback

