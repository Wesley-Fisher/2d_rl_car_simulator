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

        v0 = net.model(s0).value
        
        states = [s0]
        
        # Positive Change
        data = []
        for state in states:
            dat = net.make_dummy_data()
            net.state = state
            output = net.model(s0)
            dat.target = [output.force.action, output.angle.action, output.value + 1.0]
            dat.advantage = [dat.target[2]]
            data.append(dat)

        net.fit_model(data)

        v1 = net.model(s0).value
        self.assertGreater(float(v1), float(v0))

        v0 = v1
        # Negative Change
        data = []
        for state in states:
            dat = net.make_dummy_data()
            net.state = state
            output = net.model(s0)
            #dat.target = [output.force.action, output.angle.action, output.value - 1.0]
            dat.force = output.force.get_action_int()
            dat.angle = output.angle.get_action_int()
            dat.ret = output.value - 1.0
            dat.advantage = [dat.ret]
            data.append(dat)

        net.fit_model(data)
        v1 = net.model(s0).value
        self.assertLess(float(v1),float(v0))


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

        afi = net.model(s0).force.get_action_index()
        af0 = net.model(s0).force.action[afi]

        aai = net.model(s0).angle.get_action_index()
        aa0 = net.model(s0).angle.action[aai]
        states = [s0]

        # Increase likelihoods
        data = []
        for state in states:
            dat = net.make_dummy_data()
            dat.state = state
            output = net.model(s0)

            # Pretend that: 
            #  - 'actual' force used was greater than current
            #  - 'actual' angle used was lower than current
            #  - advantage high
            # Should see actions be more probable
            #dat.target = [output.force.action + 0.5, output.angle.action - 0.5, output.value]
            dat.force = output.force.get_action_int()
            dat.angle = output.angle.get_action_int()
            dat.ret = output.value
            dat.advantage = [5.0]
            data.append(dat)

        net.fit_model(data)
        af1 = net.model(s0).force.action[afi]
        aa1 = net.model(s0).angle.action[aai]

        self.assertGreater(float(af1), float(af0))
        self.assertGreater(float(aa1), float(aa0))


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

        afi = net.model(s0).force.get_action_index()
        af0 = net.model(s0).force.action[afi]

        aai = net.model(s0).force.get_action_index()
        aa0 = net.model(s0).angle.action[aai]
        states = [s0]

        # Increase likelihoods
        data = []
        for state in states:
            dat = net.make_dummy_data()
            dat.state = state
            output = net.model(s0)

            # Pretend that: 
            #  - 'actual' force used was greater than current
            #  - 'actual' angle used was lower than current
            #  - advantage low
            # Should see actions be less probable
            dat.force = output.force.get_action_int()
            dat.angle = output.angle.get_action_int()
            #dat.target = [output.force.action + 0.5, output.angle.action - 0.5, output.value]
            dat.ret = output.value
            dat.advantage = [-5.0]
            data.append(dat)

        net.fit_model(data)
        af1 = net.model(s0).force.action[afi]
        aa1 = net.model(s0).angle.action[aai]

        self.assertLess(float(af1), float(af0))
        self.assertLess(float(aa1), float(aa0))

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

        v0 = net.model(s0).value

        states = [s0]

        for i in range(0, 25):
            data = []
            for state in states:
                dat = net.make_dummy_data()
                dat.state = state
                output = net.model(s0)
                dat.force = output.force.get_action_int()
                dat.angle = output.angle.get_action_int()
                #dat.target = [output.force.action, output.angle.action, output.value + 1.0]
                dat.ret = output.value + 1.0
                dat.advantage = [output.value]
                data.append(dat)

            net.fit_model(data)

            v1 = net.model(s0).value
            self.assertGreater(float(v1), float(v0))


if __name__ == '__main__':
    unittest.main()