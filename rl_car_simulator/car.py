
import numpy as np

from .utilities import Utility

class CarState:
    def __init__(self):
        self.x = 0.0
        self.vx = 0.0

        self.y = 0.0
        self.vy = 0.0

        self.h = 0.0
        self.vh = 0.0

class CarControls:
    def __init__(self, force, steer):
        self.force = force
        self.steer = steer

class Car:
    def __init__(self, settings, state):
        self.settings = settings
        self.state = state
        self.controls = CarControls(0.0, 0.0)

        self.util = Utility()
        self.episode_steps = []
        self.step_experience = None
        self.goal = (0,0)
        self.goal_id = -1

        self.sensed_state = None

        self.collided = False
        self.reached_goal = True

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
            corner = c + np.dot(R, delta)
            corners.append(corner)
        return corners

    def set_controls(self, controls):
        self.controls = controls

class CarStepExperience:
    def __init__(self):
        self.s0 = None
        self.a0 = None
        self.r1 = None
        self.s1 = None
    
    def set_s0(self, s):
        self.s0 = s
    
    def set_a0(self, a):
        self.a0 = a
    
    def set_r1(self, r):
        self.r1 = r
    
    def set_s1(self, s):
        self.s1 = s