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
        advantages = []
        for state in states:
            s = np.array(state).reshape((1,len(state)))
            target = net.model(s)[0]
            #print(target)
            target = np.array([[float(target[0])], [float(target[1])], [float(target[2])]])
            targets.append(target)
            advantages.append(float(target[2]) + 1.0)

        net.fit_model(states, targets, advantages)

        v1 = net.model(s_net)[0][2]
        self.assertTrue(float(v1) - float(v0) > 0.0)

        v0 = v1
        # Negative Change
        targets = []
        for state in states:
            s = np.array(state).reshape((1,len(state)))
            target = net.model(s)[0]
            #print(target)
            target = np.array([[float(target[0])], [float(target[1])], [float(target[2])]])
            targets.append(target)
            advantages.append(float(target[2]) - 2.5) # May need to be a bigger - than the prev is a +?

        net.fit_model(states, targets, advantages)
        v1 = net.model(s_net)[0][2]
        self.assertTrue(float(v1) - float(v0) < 0.0)


    def test_update_weights_actor_good(self):
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
        af0 = net.model(s_net)[0][0]
        aa0 = net.model(s_net)[0][1]
        states = [s0]

        # Increase likelihoods
        targets = []
        advantages = []
        for state in states:
            s = np.array(state).reshape((1,len(state)))
            target = net.model(s)[0]
            #print(target)
            pred_force_0 = float(target[0])
            pred_angle_0 = float(target[1])

            # Pretend that: 
            #  - 'actual' force used was greater than current
            #  - 'actual' angle used was lower than current
            #  - advantage high
            # Should see actions be more probable
            target = np.array([[pred_force_0 + 0.5], [pred_angle_0 - 0.5], [float(target[2])]])
            targets.append(target)
            advantages.append(float(target[2]) + 5.0)

        net.fit_model(states, targets, advantages)
        af1 = net.model(s_net)[0][0]
        aa1 = net.model(s_net)[0][1]

        self.assertGreater(float(af1), float(af0))
        self.assertLess(float(aa1), float(aa0))


    def test_update_weights_actor_bad(self):
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
        af0 = net.model(s_net)[0][0]
        aa0 = net.model(s_net)[0][1]
        states = [s0]

        # Decrease likelihoods
        targets = []
        advantages = []
        for state in states:
            s = np.array(state).reshape((1,len(state)))
            target = net.model(s)[0]
            #print(target)
            pred_force_0 = float(target[0])
            pred_angle_0 = float(target[1])

            # Pretend that: 
            #  - 'actual' force used was greater than current
            #  - 'actual' angle used was lower than current
            #  - advantage low
            # Should see actions be less probable
            target = np.array([[pred_force_0 + 0.5], [pred_angle_0 - 0.5], [float(target[2])]])
            targets.append(target)
            advantages.append(float(target[2]) - 5.0)

        net.fit_model(states, targets, advantages)
        af1 = net.model(s_net)[0][0]
        aa1 = net.model(s_net)[0][1]

        self.assertLess(float(af1), float(af0))
        self.assertGreater(float(aa1), float(aa0))

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
            advantages = []
            for state in states:
                s = np.array(state).reshape((1,len(state)))
                target = np.array(net.model(s)[0])
                #print(target)
                target = np.array([[float(target[0])], [float(target[1])], [float(target[2])]])
                targets.append(target)
                advantages.append(float(target[2]) + 1.0)

            net.fit_model(states, targets, advantages)

            v1 = net.model(s_net)[0][2]
            self.assertTrue(float(v1) - float(v0) > 0.0)


if __name__ == '__main__':
    unittest.main()