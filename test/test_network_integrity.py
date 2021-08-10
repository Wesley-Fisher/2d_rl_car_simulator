import math
import numpy as np
import os
import time

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
        print(net._model._model.summary())
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
        experience_file = dir_path + "/temp_data/memory/experience.pk"
        model_file = dir_path + "/temp_data/memory/model.h5"
        settings._files.root_dir = dir_path + "/temp_data"
        settings.memory.load_saved_network = True
        settings.memory.load_saved_experience = True

        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        cs = CarState()
        car = Car(settings, cs)
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))

        s_net = np.array(s0).reshape((1,len(s0)))

        os.remove(experience_file)
        os.remove(model_file)
        time.sleep(0.1)

        net.save_state()
        time.sleep(0.1)
        self.assertTrue(os.path.isfile(experience_file))
        self.assertTrue(os.path.isfile(model_file))
        self.assertTrue(net.load_state())

        os.remove(experience_file)
        os.remove(model_file)
        time.sleep(0.1)

        v0 = net.get(s_net)[0][2]
        v0 = net.model(s_net)[0][2]
        v0 = net.model(s_net, net._model)[0][2]
        v0 = net.model(s_net, net.frozen_model)[0][2]
        self.assertTrue(True)

    def test_freeze(self):
        settings = Settings()
        settings.memory.load_saved_network = False
        settings.memory.load_saved_experience = False

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
        experience_file = dir_path + "/temp_data/memory/experience.pk"
        model_file = dir_path + "/temp_data/memory/model.h5"
        settings._files.root_dir = dir_path + "/temp_data"
        settings.memory.load_saved_network = True
        settings.memory.load_saved_experience = True

        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        cs = CarState()
        car = Car(settings, cs)
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))

        s_net = np.array(s0).reshape((1,len(s0)))

        os.remove(experience_file)
        os.remove(model_file)
        time.sleep(0.1)

        net.save_state()
        time.sleep(0.1)

        self.assertTrue(os.path.isfile(experience_file))
        self.assertTrue(os.path.isfile(model_file))
        self.assertTrue(net.load_state())
        net.freeze()

        os.remove(experience_file)
        os.remove(model_file)
        time.sleep(0.1)

        v0 = net.get(s_net)[0][2]
        v0 = net.model(s_net)[0][2]
        v0 = net.model(s_net, net._model)[0][2]
        v0 = net.model(s_net, net.frozen_model)[0][2]
        self.assertTrue(True)
    
    def test_save_load_freeze_repeated(self):
        settings = Settings()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        experience_file = dir_path + "/temp_data/memory/experience.pk"
        model_file = dir_path + "/temp_data/memory/model.h5"
        settings._files.root_dir = dir_path + "/temp_data"
        settings.memory.load_saved_network = True
        settings.memory.load_saved_experience = True

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
            os.remove(experience_file)
            os.remove(model_file)
            time.sleep(0.1)

            net.save_state()
            time.sleep(0.1)

            self.assertTrue(os.path.isfile(experience_file))
            self.assertTrue(os.path.isfile(model_file))
            net.load_state()
            net.freeze()

            v0 = net.get(s_net)[0][2]
            v0 = net.model(s_net)[0][2]
            v0 = net.model(s_net, net._model)[0][2]
            v0 = net.model(s_net, net.frozen_model)[0][2]
            self.assertTrue(True)

    def test_failed_load_freeze_repeated(self):
        settings = Settings()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        experience_file = dir_path + "/temp_data/memory/experience.pk"
        model_file = dir_path + "/temp_data/memory/model.h5"
        settings._files.root_dir = dir_path + "/temp_data"
        settings.memory.load_saved_network = True
        settings.memory.load_saved_experience = True

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
            try:
                os.remove(experience_file)
            except Error as e:
                pass

            try:
                os.remove(model_file)
            except Error as e:
                pass
            time.sleep(0.1)

            self.assertFalse(os.path.isfile(experience_file))
            self.assertFalse(os.path.isfile(model_file))
            self.assertFalse(net.load_state())
            net.freeze()

            v0 = net.get(s_net)[0][2]
            v0 = net.model(s_net)[0][2]
            v0 = net.model(s_net, net._model)[0][2]
            v0 = net.model(s_net, net.frozen_model)[0][2]
            self.assertTrue(True)                


if __name__ == '__main__':
    unittest.main()