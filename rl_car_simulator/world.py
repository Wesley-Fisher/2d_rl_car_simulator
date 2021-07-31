import random

from .controllers import ControllerTypes

class World:
    def __init__(self, settings):
        self.settings = settings

        self.all_cars = []
        self.keyboard_cars = []
        self.network_cars = []
        self.hardcoded_cars = []
        self.random_cars = []
        self.feedback_cars = []

        self.walls = []

    def add_keyboard_car(self, car):
        self.all_cars.append(car)
        self.keyboard_cars.append(car)
        car.set_type(ControllerTypes.keyboard)
        car.set_name(ControllerTypes.keyboard + " " + str(len(self.keyboard_cars)))
    
    def add_network_car(self, car):
        self.all_cars.append(car)
        self.network_cars.append(car)
        car.set_type(ControllerTypes.network)
        car.set_name(ControllerTypes.network + " " + str(len(self.keyboard_cars)))

    def add_random_car(self, car):
        self.all_cars.append(car)
        self.random_cars.append(car)
        car.set_type(ControllerTypes.random)
        car.set_name(ControllerTypes.random + " " + str(len(self.keyboard_cars)))

    def add_feedback_car(self, car):
        self.all_cars.append(car)
        self.feedback_cars.append(car)
        car.set_type(ControllerTypes.feedback)
        car.set_name(ControllerTypes.feedback + " " + str(len(self.keyboard_cars)))

    def add_wall(self, wall):
        self.walls.append(wall)
    
    def random_spawn_state(self):
        p = random.choice(self.settings.world.spawn_points)
        h = random.uniform(-3.14, 3.14)
        return (p[0],p[1],h)

    def random_goal_state(self):
        id = random.randint(0, len(self.settings.world.goal_points)-1)
        return self.settings.world.goal_points[id], id