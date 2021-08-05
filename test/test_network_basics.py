import math
import numpy as np

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

    def test_update_weights_critic(self):
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
        v0 = net.model(s_net)[0][2]
        
        states = [s0]
        
        # Positive Change
        targets = []
        for state in states:
            s = np.array(state).reshape((1,len(state)))
            target = np.array(net.model(s)[0])
            print(target)
            target[2] = target[2] + 0.5
            targets.append(target)

        net.fit_model(states, targets)

        v1 = net.model(s_net)[0][2]
        self.assertTrue(float(v1) - float(v0) > 0.0)

        v0 = v1
        # Negative Change
        targets = []
        for state in states:
            s = np.array(state).reshape((1,len(state)))
            target = np.array(net.model(s)[0])
            print(target)
            target[2] = target[2] - 0.5
            targets.append(target)

        net.fit_model(states, targets)
        v1 = net.model(s_net)[0][2]
        self.assertTrue(float(v1) - float(v0) < 0.0)


    def test_update_weights_actor(self):
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
        a0 = net.model(s_net)[0][0]

        states = [s0]

        # Positive Change
        targets = []
        for state in states:
            s = np.array(state).reshape((1,len(state)))
            target = np.array(net.model(s)[0])
            print(target)
            target[0] = target[0] + 0.5
            targets.append(target)

        net.fit_model(states, targets)

        a1 = net.model(s_net)[0][0]
        self.assertTrue(float(a1) - float(a0) > 0.0)

        a0 = a1
        # Negative Change
        targets = []
        for state in states:
            s = np.array(state).reshape((1,len(state)))
            target = np.array(net.model(s)[0])
            print(target)
            target[0] = target[0] - 0.5
            targets.append(target)

        net.fit_model(states, targets)

        a1 = net.model(s_net)[0][0]
        self.assertTrue(float(a1) - float(a0) < 0.0)


    def test_gradient_ascent_critic(self):
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
        v0 = net.model(s_net)[0][2]

        states = [s0]

        for i in range(0, 25):
            targets = []
            for state in states:
                s = np.array(state).reshape((1,len(state)))
                target = np.array(net.model(s)[0])
                print(target)
                target[2] = target[2] + 0.5
                targets.append(target)

            net.fit_model(states, targets)

            v1 = net.model(s_net)[0][2]
            self.assertTrue(float(v1) - float(v0) > 0.0)


if __name__ == '__main__':
    unittest.main()