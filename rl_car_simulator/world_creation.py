import random

from .world import World
from .car import Car, CarState
from .walls import Wall

class WorldCreation:
    def __init__(self, settings):
        self.world = World(settings)

        def make_car():
            (x,y,h) = self.world.random_spawn_state()
            cs = CarState()
            cs.x = x
            cs.y = y
            cs.h = h
            c = Car(settings, cs)
            goal, id = self.world.random_goal_state()
            c.set_goal(goal, id)
            return c

        for i in range(0, min(settings.initial_car_settings.keyboard_cars,1)):
            self.world.add_keyboard_car(make_car())

        for i in range(0, settings.initial_car_settings.network_cars):
            self.world.add_network_car(make_car())

        for i in range(0, settings.initial_car_settings.random_cars):
            self.world.add_random_car(make_car())

        for i in range(0, settings.initial_car_settings.feedback_cars):
            self.world.add_feedback_car(make_car())

        for i in range(0, settings.initial_car_settings.network_exploration_cars):
            self.world.add_network_exploration_car(make_car())
        
        for i in range(0, settings.initial_car_settings.feedback_exploration_cars):
            self.world.add_feedback_exploration_car(make_car())

        for wall_pts in settings.walls.walls:
            w = Wall(wall_pts)
            self.world.add_wall(w)

    def get(self):
        return self.world
