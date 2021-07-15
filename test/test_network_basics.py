import math
import numpy as np

import unittest

from rl_car_simulator.settings import Settings
from rl_car_simulator.network import Network
from rl_car_simulator.physics_engine import PhysicsEngine
from rl_car_simulator.world import World
from rl_car_simulator.world_creation import WorldCreation
from rl_car_simulator.car import Car, CarState, CarStepExperience


class TestNetworkBasics(unittest.TestCase):

    def test_smoke(self):
        settings = Settings()
        world = WorldCreation(settings).get()
        physics = PhysicsEngine(settings, world)
        null_state = physics.get_car_state(Car(settings, CarState()))
        net = Network(settings, len(null_state))
        self.assertTrue(True)

    def test_update_weights_critic(self):
        settings = Settings()
        world = WorldCreation(settings).get()
        physics = PhysicsEngine(settings, world)
        cs = CarState()
        car = Car(settings, cs)
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))

        s0 = np.array(s0).reshape((1,len(s0)))
        
        # Positive Change
        v0, gradient_critic, trainable_critic, _, _, _ = net.calculate_gradients(s0)
        net.update_weights(1e-3, gradient_critic, trainable_critic)
        v1 = net.model(s0)[0][2]
        self.assertTrue(float(v1) - float(v0) > 0.0)

        # Negative Change
        v0, gradient_critic, trainable_critic, _, _, _ = net.calculate_gradients(s0)
        net.update_weights(-1e-3, gradient_critic, trainable_critic)
        v1 = net.model(s0)[0][2]
        self.assertTrue(float(v1) - float(v0) < 0.0)

    def test_update_weights_actor(self):
        settings = Settings()
        world = WorldCreation(settings).get()
        physics = PhysicsEngine(settings, world)
        cs = CarState()
        car = Car(settings, cs)
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))

        s0 = np.array(s0).reshape((1,len(s0)))
        
        # Positive Change
        _, _, _, a0, gradient_actor, trainable_actor = net.calculate_gradients(s0)
        net.update_weights(1e-3, gradient_actor, trainable_actor)
        a1 = net.model(s0)[0][0:2]
        self.assertTrue(float(a1[0]) - float(a0[0]) > 0.0)
        self.assertTrue(float(a1[1]) - float(a0[1]) > 0.0)

        # Negative Change
        _, _, _, a0, gradient_actor, trainable_actor = net.calculate_gradients(s0)
        net.update_weights(-1e-3, gradient_actor, trainable_actor)
        a1 = net.model(s0)[0][0:2]
        self.assertTrue(float(a1[0]) - float(a0[0]) < 0.0)
        self.assertTrue(float(a1[1]) - float(a0[1]) < 0.0)

    def test_gradient_ascent_critic(self):
        settings = Settings()
        world = WorldCreation(settings).get()
        physics = PhysicsEngine(settings, world)
        cs = CarState()
        car = Car(settings, cs)
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))

        s0 = np.array(s0).reshape((1,len(s0)))
        
        for i in range(0, 25):
            v0, gradient_critic, train_critic, _, _, _ = net.calculate_gradients(s0)
            net.update_weights(1e-4, gradient_critic, train_critic)
            v1 = net.model(s0)[0][2]
            self.assertTrue(float(v1) - float(v0) > 0.0)



if __name__ == '__main__':
    unittest.main()