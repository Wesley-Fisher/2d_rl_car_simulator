import keyboard
import random
import math

from .car import CarControls

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

class Controllers:
    def __init__(self, keyboard, network, hardcoded, random):
        self.keyboard = keyboard
        self.network = network
        self.hardcoded = hardcoded
        self.random = random

