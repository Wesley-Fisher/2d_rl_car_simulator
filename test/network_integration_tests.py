import math
import numpy as np

import unittest

from rl_car_simulator.settings import Settings
from rl_car_simulator.network import Network
from rl_car_simulator.physics_engine import PhysicsEngine
from rl_car_simulator.world import World
from rl_car_simulator.walls import Wall
from rl_car_simulator.world_creation import WorldCreation
from rl_car_simulator.car import Car, CarState, CarStepExperience
from rl_car_simulator.controllers import Controller, HardCodedController, Controllers
from rl_car_simulator.experience_preprocessor import ExperiencePreprocessor
from rl_car_simulator.experience_engine import ExperienceEngine

class TestNetworkIntegration(unittest.TestCase):

    def test_smoke(self):
        settings = Settings()
        world = WorldCreation(settings).get()
        physics = PhysicsEngine(settings, world)
        null_state = physics.get_car_state(Car(settings, CarState()))
        net = Network(settings, len(null_state))
        self.assertTrue(True)

    def generate_collision_processed_experience(self):
        settings = Settings()
        settings.physics.physics_timestep = 0.05
        settings.physics.control_timestep = 0.05
        settings.learning.alpha = 1e-4

        world = WorldCreation(settings).get()
        world.walls.append(Wall(((25,0),(25,20))))
        physics = PhysicsEngine(settings, world)
        cs = CarState()
        cs.x = 20.0
        cs.y = 10.0
        cs.h = 0.0
        car = Car(settings, cs)
        car.goal = [-5, -5]
        world.keyboard_cars = []
        world.network_cars = []
        world.hardcoded_cars = [car]
        world.all_cars = [car]
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))
        controller = HardCodedController(settings, 1.0, 0.0)
        controllers = Controllers(None, None, controller)
        physics.set_controllers(controllers)

        s0 = np.array(s0).reshape((1,len(s0)))

        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)

        # Generate one set of experience of driving into a wall
        i = 0
        done = False
        while not done:
            # Mock physics and experience loop
            # - Duplicates code, but can extract time when done
            physics.sensors_step()

            experience.sample_end_states()
            experience.sample_rewards()

            done = car.collided
            experience.handle_episode_ends()
            physics.handle_resets()

            experience.new_experience_step()

            physics.controls_step()

            experience.sample_start_states()
            experience.sample_controls()

            physics.physics_time_step()
            physics.handle_goals()
            physics.handle_collisions()

            #print(car.state.x)

            # Catch errors that lead to inf loop
            self.assertTrue(i < 1000)
            i = i + 1
        self.assertTrue(done)
        self.assertGreater(i, 10)

        experience_raw = preprocessor.experience_queue.pop(1)
        self.assertGreater(len(experience_raw), 2)

        experience = preprocessor.preprocess_episode(experience_raw)
        return experience, net

    def test_collision_learning(self):
        experience, net = self.generate_collision_processed_experience()
        
        ex0 = experience[0]
        exM = experience[int(len(experience)/2)]
        exF = experience[-1]

        v0_last = float(net.model(ex0.s0)[0][2])
        vM_last = float(net.model(exM.s0)[0][2])
        vF_last = float(net.model(exF.s0)[0][2])

        for i in range(0, 5):

            for ex in experience:
                net.train_sample(ex)
            
            v0 = float(net.model(ex0.s0)[0][2])
            vM = float(net.model(exM.s0)[0][2])
            vF = float(net.model(exF.s0)[0][2])

            self.assertLess(v0, v0_last)
            self.assertLess(vM, vM_last)
            self.assertLess(vF, vF_last)

            v0_last = v0
            vM_last = vM
            vF_last = vF

    def generate_goal_processed_experience(self):
        settings = Settings()
        settings.physics.physics_timestep = 0.05
        settings.physics.control_timestep = 0.05
        settings.learning.alpha = 1e-4

        world = WorldCreation(settings).get()
        physics = PhysicsEngine(settings, world)
        cs = CarState()
        cs.x = 20.0
        cs.y = 10.0
        cs.h = 3.14159
        car = Car(settings, cs)
        car.goal = [10.0, 10.0]
        car.reached_goal = False
        world.keyboard_cars = []
        world.network_cars = []
        world.hardcoded_cars = [car]
        world.all_cars = [car]
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))
        controller = HardCodedController(settings, 1.0, 0.0)
        controllers = Controllers(None, None, controller)
        physics.set_controllers(controllers)

        s0 = np.array(s0).reshape((1,len(s0)))

        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)

        # Generate one set of experience of driving into a wall
        i = 0
        done = False
        while not done:
            # Mock physics and experience loop
            # - Duplicates code, but can extract time when done
            physics.sensors_step()

            experience.sample_end_states()
            experience.sample_rewards()

            done = car.reached_goal
            experience.handle_episode_ends()
            physics.handle_resets()

            experience.new_experience_step()

            physics.controls_step()

            experience.sample_start_states()
            experience.sample_controls()

            physics.physics_time_step()
            physics.handle_goals()
            physics.handle_collisions()

            #print(car.state.x)

            # Catch errors that lead to inf loop
            self.assertTrue(i < 1000)
            i = i + 1
        self.assertTrue(done)
        self.assertGreater(i, 10)

        experience_raw = preprocessor.experience_queue.pop(1)
        self.assertGreater(len(experience_raw), 2)

        experience = preprocessor.preprocess_episode(experience_raw)
        return experience, net

    def test_goal_learning(self):
        experience, net = self.generate_goal_processed_experience()
        
        ex0 = experience[0]
        exM = experience[int(len(experience)/2)]
        exF = experience[-1]

        v0_last = float(net.model(ex0.s0)[0][2])
        vM_last = float(net.model(exM.s0)[0][2])
        vF_last = float(net.model(exF.s0)[0][2])

        for i in range(0, 5):

            for ex in experience:
                net.train_sample(ex)
            
            v0 = float(net.model(ex0.s0)[0][2])
            vM = float(net.model(exM.s0)[0][2])
            vF = float(net.model(exF.s0)[0][2])

            self.assertGreater(v0, v0_last)
            self.assertGreater(vM, vM_last)
            self.assertGreater(vF, vF_last)

            v0_last = v0
            vM_last = vM
            vF_last = vF


if __name__ == '__main__':
    test = TestNetworkBasics()
    test.test_smoke()