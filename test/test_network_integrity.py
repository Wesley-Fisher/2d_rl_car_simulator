import math
import numpy as np
import os

import unittest

from rl_car_simulator.settings import Settings
from rl_car_simulator.network import Network
from rl_car_simulator.physics_engine import PhysicsEngine
from rl_car_simulator.world import World
from rl_car_simulator.world_creation import WorldCreation
from rl_car_simulator.car import Car, CarState, CarStepExperience
from rl_car_simulator.experience_preprocessor import ExperiencePreprocessor
from rl_car_simulator.experience_engine import ExperienceEngine

class TestNetworkBasics(unittest.TestCase):

    def test_smoke(self):
        settings = Settings()
        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        null_state = physics.get_car_state(Car(settings, CarState()))
        net = Network(settings, len(null_state))
        self.assertTrue(True)

    def test_model_get_interfaces(self):
        settings = Settings()
        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        cs = CarState()
        car = Car(settings, cs)
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))

        s_net = np.array(s0).reshape((1,len(s0)))

        v0 = net.get(s_net)[0][2]
        v0 = net.model(s_net)[0][2]
        v0 = net.model(s_net, net._model)[0][2]
        v0 = net.model(s_net, net.frozen_model)[0][2]
        self.assertTrue(True)

    def test_save_load(self):
        settings = Settings()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        settings._files.root_dir = dir_path + "/temp_data"

        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        cs = CarState()
        car = Car(settings, cs)
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))

        s_net = np.array(s0).reshape((1,len(s0)))

        net.save_state()
        net.load_state()

        os.remove(dir_path + "/temp_data/memory/experience.pk")
        os.remove(dir_path + "/temp_data/memory/model.h5")

        v0 = net.get(s_net)[0][2]
        v0 = net.model(s_net)[0][2]
        v0 = net.model(s_net, net._model)[0][2]
        v0 = net.model(s_net, net.frozen_model)[0][2]
        self.assertTrue(True)

    def test_freeze(self):
        settings = Settings()
        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        cs = CarState()
        car = Car(settings, cs)
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))

        s_net = np.array(s0).reshape((1,len(s0)))
        net.freeze()

        v0 = net.get(s_net)[0][2]
        v0 = net.model(s_net)[0][2]
        v0 = net.model(s_net, net._model)[0][2]
        v0 = net.model(s_net, net.frozen_model)[0][2]

    def test_save_load_freeze(self):
        settings = Settings()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        settings._files.root_dir = dir_path + "/temp_data"

        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        cs = CarState()
        car = Car(settings, cs)
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))

        s_net = np.array(s0).reshape((1,len(s0)))

        net.save_state()
        net.load_state()
        net.freeze()

        os.remove(dir_path + "/temp_data/memory/experience.pk")
        os.remove(dir_path + "/temp_data/memory/model.h5")

        v0 = net.get(s_net)[0][2]
        v0 = net.model(s_net)[0][2]
        v0 = net.model(s_net, net._model)[0][2]
        v0 = net.model(s_net, net.frozen_model)[0][2]
        self.assertTrue(True)
    
    def test_save_load_freeze_repeated(self):
        settings = Settings()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        settings._files.root_dir = dir_path + "/temp_data"

        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        cs = CarState()
        car = Car(settings, cs)
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))

        s_net = np.array(s0).reshape((1,len(s0)))
        for i in range(0, 3):
            net.save_state()
            net.load_state()
            net.freeze()

            os.remove(dir_path + "/temp_data/memory/experience.pk")
            os.remove(dir_path + "/temp_data/memory/model.h5")

            v0 = net.get(s_net)[0][2]
            v0 = net.model(s_net)[0][2]
            v0 = net.model(s_net, net._model)[0][2]
            v0 = net.model(s_net, net.frozen_model)[0][2]
            self.assertTrue(True)

                


if __name__ == '__main__':
    unittest.main()