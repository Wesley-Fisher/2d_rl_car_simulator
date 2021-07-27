import math

import unittest

from rl_car_simulator.walls import Wall
from rl_car_simulator.car import Car, CarState
from rl_car_simulator.settings import Settings
from rl_car_simulator.world import World
from rl_car_simulator.physics_engine import PhysicsEngine
from rl_car_simulator.experience_preprocessor import ExperiencePreprocessor
from rl_car_simulator.experience_engine import ExperienceEngine


class TestLidarBasic(unittest.TestCase):

    def test_smoke(self):
        settings = Settings()
        world = World(settings)
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        pe = PhysicsEngine(settings, world, experience)
        car_state = CarState()
        car = Car(settings, car_state)

    def test_lidar_flat_simple(self):
        settings = Settings()
        world = World(settings)
        world.walls.append(Wall(((10,-5),(10,5))))
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        pe = PhysicsEngine(settings, world, experience)
        car_state = CarState()
        car_state.x = 0.0
        car_state.y = 0.0
        car_state.h = 0.0
        car = Car(settings, car_state)

        d = pe.calc_lidar_distance(car, 0.0)
        self.assertAlmostEqual(d, 10.0)

    def test_lidar_flat_double(self):
        settings = Settings()
        world = World(settings)
        world.walls.append(Wall(((10,-5),(10,5))))
        world.walls.append(Wall(((11,-5),(11,5))))
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        pe = PhysicsEngine(settings, world, experience)
        car_state = CarState()
        car_state.x = 0.0
        car_state.y = 0.0
        car_state.h = 0.0
        car = Car(settings, car_state)

        d = pe.calc_lidar_distance(car, 0.0)
        self.assertAlmostEqual(d, 10.0)
    
    def test_lidar_flat_behind(self):
        settings = Settings()
        world = World(settings)
        world.walls.append(Wall(((10,-5),(10,5))))
        world.walls.append(Wall(((-5,-5),(-5,5))))
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        pe = PhysicsEngine(settings, world, experience)
        car_state = CarState()
        car_state.x = 0.0
        car_state.y = 0.0
        car_state.h = 0.0
        car = Car(settings, car_state)

        d = pe.calc_lidar_distance(car, 0.0)
        self.assertAlmostEqual(d, 10.0)

    def test_lidar_angled(self):
        settings = Settings()
        world = World(settings)
        world.walls.append(Wall(((8,-5),(12,5))))
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        pe = PhysicsEngine(settings, world, experience)
        car_state = CarState()
        car_state.x = 0.0
        car_state.y = 0.0
        car_state.h = 0.0
        car = Car(settings, car_state)

        d = pe.calc_lidar_distance(car, 0.0)
        self.assertAlmostEqual(d, 10.0)

    def test_lidar_heading(self):
        settings = Settings()
        world = World(settings)
        L = 10.0
        ang = 0.2
        world.walls.append(Wall(((L,-5),(L,5))))
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        pe = PhysicsEngine(settings, world, experience)
        car_state = CarState()
        car_state.x = 0.0
        car_state.y = 0.0
        car_state.h = ang
        car = Car(settings, car_state)

        d = pe.calc_lidar_distance(car, 0.0)
        self.assertAlmostEqual(d, L/math.cos(ang))

    def test_lidar_heading_angled(self):
        settings = Settings()
        world = World(settings)
        L = 10.0
        ang = 0.2
        world.walls.append(Wall(((L,-5),(L,5))))
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        pe = PhysicsEngine(settings, world, experience)
        car_state = CarState()
        car_state.x = 0.0
        car_state.y = 0.0
        car_state.h = ang
        car = Car(settings, car_state)

        d = pe.calc_lidar_distance(car, ang)
        self.assertAlmostEqual(d, L/math.cos(2*ang))



if __name__ == '__main__':
    unittest.main()