import keyboard

from .car import CarControls

class Controller:
    def __init__(self, settings):
        pass

    def get_controls(self, state):
        raise NotImplementedError


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
        force = min(max(control[0][0], -self.settings.keyboard.force), self.settings.keyboard.force)
        angle = min(max(control[0][1], -self.settings.keyboard.angle), self.settings.keyboard.angle)
        return CarControls(force, angle)

class Controllers:
    def __init__(self, keyboard, network):
        self.keyboard = keyboard
        self.network = network

