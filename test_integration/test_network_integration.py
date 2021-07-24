import math
import numpy as np
import copy

import tensorflow as tf

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

    coll_exp_1 = None
    goal_exp_2 = None
    network_1 = None

    @classmethod
    def setUpClass(cls):
        settings = Settings()
        settings.learning.max_episode_length = 200
        settings.physics.physics_timestep = 0.2
        settings.physics.control_timestep = 0.2
        settings.learning.alpha = 1e-4
        cls.coll_exp_1, cls.network_1 = cls.generate_collision_processed_experience(settings)
        cls.goal_exp_1,             _ = cls.generate_goal_processed_experience(settings)
        cls.coll_exp_2,             _ = cls.generate_collision_processed_experience_turning(settings)
        cls.goal_exp_2,             _ = cls.generate_goal_processed_experience_turning(settings)

    def test_smoke(self):
        settings = Settings()
        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        null_state = physics.get_car_state(Car(settings, CarState()))
        net = Network(settings, len(null_state))
        self.assertTrue(True)

    def generate_collision_processed_experience(settings):
        world = WorldCreation(settings).get()
        world.walls.append(Wall(((25,0),(25,20))))
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
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



        # Generate one set of experience of driving into a wall
        i = 0
        while len(preprocessor.experience_queue) < 2:
            physics.full_control_sensor_step()
            physics.full_physics_termination_step()

            # Catch errors that lead to inf loop
            assert(i < 1000)
            i = i + 1
        assert(i > 10)

        experience_raw = preprocessor.experience_queue.pop(1)
        #print([ex.r1 for ex in experience_raw])
        assert(len(experience_raw) > 2)

        experience = preprocessor.preprocess_episode(experience_raw)
        return experience, net

    def generate_collision_processed_experience_turning(settings):
        world = WorldCreation(settings).get()
        world.walls.append(Wall(((20,15),(40,15))))
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
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
        controller = HardCodedController(settings, 1.0, 0.3)
        controllers = Controllers(None, None, controller)
        physics.set_controllers(controllers)

        s0 = np.array(s0).reshape((1,len(s0)))


        # Generate one set of experience of driving into a wall
        i = 0
        while len(preprocessor.experience_queue) < 2:
            physics.full_control_sensor_step()
            physics.full_physics_termination_step()

            #print("%f, %f" % (car.state.x, car.state.y))
            # Catch errors that lead to inf loop
            assert(i < 1000)
            i = i + 1
        assert(i > 10)

        experience_raw = preprocessor.experience_queue.pop(1)
        #print([ex.r1 for ex in experience_raw])
        assert(len(experience_raw) > 2)

        experience = preprocessor.preprocess_episode(experience_raw)
        return experience, net

    def generate_goal_processed_experience(settings):
        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        cs = CarState()
        cs.x = 20.0
        cs.y = 10.0
        cs.h = 3.14159
        car = Car(settings, cs)
        car.goal = [15.0, 10.0]
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

        # Generate one set of experience of driving into a wall
        i = 0
        while len(preprocessor.experience_queue) < 1:
            physics.full_control_sensor_step()
            physics.full_physics_termination_step()

            # Catch errors that lead to inf loop
            assert(i < 1000)
            i = i + 1
        assert(i > 10)

        experience_raw = preprocessor.experience_queue.pop(0)
        assert(len(experience_raw) > 2)

        experience = preprocessor.preprocess_episode(experience_raw)
        return experience, net

    def generate_goal_processed_experience_turning(settings):
        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        cs = CarState()
        cs.x = 20.0
        cs.y = 10.0
        cs.h = 0.0
        car = Car(settings, cs)
        car.goal = [25.0, 5.0]
        car.reached_goal = False
        world.keyboard_cars = []
        world.network_cars = []
        world.hardcoded_cars = [car]
        world.all_cars = [car]
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))
        controller = HardCodedController(settings, 1.0, -0.2)
        controllers = Controllers(None, None, controller)
        physics.set_controllers(controllers)

        s0 = np.array(s0).reshape((1,len(s0)))

        # Generate one set of experience of driving into a wall
        i = 0
        while len(preprocessor.experience_queue) < 1:
            physics.full_control_sensor_step()
            physics.full_physics_termination_step()

            #print("%f, %f" % (car.state.x, car.state.y))
            # Catch errors that lead to inf loop
            assert(i < 1000)
            i = i + 1
        assert(i > 10)

        experience_raw = preprocessor.experience_queue.pop(0)
        assert(len(experience_raw) > 2)

        experience = preprocessor.preprocess_episode(experience_raw)
        return experience, net

    def test_collision_learning(self):
        return
        experience = copy.deepcopy(self.coll_exp_1)
        net = copy.deepcopy(self.network_1)
        net.model = tf.keras.models.clone_model(self.network_1.model)
        
        ex0 = experience[0]
        exM = experience[int(len(experience)/2)]
        exF = experience[-1]
        try:
            v0_last = float(net.model(ex0.s0)[0][2])
        except AssertionError as e:
            print(e)
            vM_last = float(net.model(exM.s0)[0][2])
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

   
    def test_goal_learning(self):
        return
        experience = copy.deepcopy(self.goal_exp_1)
        net = copy.deepcopy(self.network_1)
        net.model = tf.keras.models.clone_model(self.network_1.model)
        
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

    def test_terminal_goal_learning(self):
        return
        settings = Settings()
        settings.learning.alpha = 1e-3
        settings.learning.gamma = 0.2
        expGoal = copy.deepcopy(self.goal_exp_1)
        net = copy.deepcopy(self.network_1)
        net.settings = settings
        net.model = tf.keras.models.clone_model(self.network_1.model)

        def print_exp(header, exp, head):
            if head:
                print(header + ":")
                print("v0\tv1\tr\td")
            for ex in exp:
                v0 = float(net.model(ex.s0)[0][2])
                v1 = float(net.model(ex.s1)[0][2])
                r = ex.r1
                d = r + settings.learning.gamma * v1 - v0
                print("%.3f\t%.3f\t%.3f\t%.3f" % (v0, v1, r, d))

        ex = expGoal[0]
        #print_exp("Goal 0", [ex], True)
        for i in range(0, 500):
            net.train_sample(ex)
            #print_exp("", [ex], False)

        v0 = float(net.model(ex.s0)[0][2])
        v1 = float(net.model(ex.s1)[0][2])
        r = ex.r1
        self.assertLess(abs(v0), abs(v1 + r))
        return

    def test_terminal_coll_learning(self):
        return
        settings = Settings()
        settings.learning.alpha = 1e-3
        settings.learning.gamma = 0.2
        expColl = copy.deepcopy(self.coll_exp_1)
        net = copy.deepcopy(self.network_1)
        net.settings = settings
        net.model = tf.keras.models.clone_model(self.network_1.model)

        def print_exp(header, exp, head):
            if head:
                print(header + ":")
                print("v0\tv1\tr\td")
            for ex in exp:
                v0 = float(net.model(ex.s0)[0][2])
                v1 = float(net.model(ex.s1)[0][2])
                r = ex.r1
                d = r + settings.learning.gamma * v1 - v0
                print("%.3f\t%.3f\t%.3f\t%.3f" % (v0, v1, r, d))

        ex = expColl[0]
        #print_exp("Goal 0", [ex], True)
        for i in range(0, 500):
            net.train_sample(ex)
            #print_exp("", [ex], False)

        v0 = float(net.model(ex.s0)[0][2])
        v1 = float(net.model(ex.s1)[0][2])
        r = ex.r1
        self.assertLess(abs(v0), abs(v1 + r))
        return

    def test_split_experience_learning(self):
        return
        settings = Settings()
        settings.learning.alpha = 1e-3
        settings.learning.gamma = 0.9
        expGoal = copy.deepcopy(self.goal_exp_1)
        expColl = copy.deepcopy(self.coll_exp_1)
        net = copy.deepcopy(self.network_1)
        net.settings = settings
        net.model = tf.keras.models.clone_model(self.network_1.model)
        
        exG0 = expGoal[0]
        exGF = expGoal[-1]
        exC0 = expColl[0]
        exCF = expColl[-1]

        expColl.reverse()
        all_exp = expGoal + expColl

        vG0_last = float(net.model(exG0.s0)[0][2])
        vGF_last = float(net.model(exGF.s0)[0][2])
        vC0_last = float(net.model(exC0.s0)[0][2])
        vCF_last = float(net.model(exCF.s0)[0][2])

        def print_exp(header, exp, head):
            if head:
                print(header + ":")
                print("v0\tv1\tr\td")
            for ex in exp:
                v0 = float(net.model(ex.s0)[0][2])
                v1 = float(net.model(ex.s1)[0][2])
                r = ex.r1
                d = r + settings.learning.gamma * v1 - v0
                print("%.3f\t%.3f\t%.3f\t%.3f" % (v0, v1, r, d))

        # Look for overall improvement in 5 iterations
        # of a few steps each
        for j in range(0, 5):
            #print("Split round training %d" % (j+1))
            for i in range(0, 20):
                for ex in all_exp:
                    net.train_sample(ex)

            #print_exp("Vals",all_exp, True)

            # Can't be as sure with training with both sets
            # So only test final results
            vG0 = float(net.model(exG0.s0)[0][2])
            vGF = float(net.model(exGF.s0)[0][2])
            vC0 = float(net.model(exC0.s0)[0][2])
            vCF = float(net.model(exCF.s0)[0][2])

        # Final States should have clear learning
        self.assertGreater(vGF, vGF_last)
        self.assertLess(vCF, vCF_last)

        self.assertLess(vC0, vC0_last)
        self.assertGreater(vG0, vG0_last)


    def test_split_experience_actor_turning(self):
        settings = Settings()
        settings.learning.alpha = 1e-3
        settings.learning.gamma = 0.9
        expGoal = copy.deepcopy(self.goal_exp_2)
        expColl = copy.deepcopy(self.coll_exp_2)
        net = copy.deepcopy(self.network_1)
        net.settings = settings
        net.model = tf.keras.models.clone_model(self.network_1.model)

        AG = -0.2
        AC = 0.3
        
        exG0 = expGoal[0]
        exGF = expGoal[-1]
        exC0 = expColl[0]
        exCF = expColl[-1]

        expColl.reverse()
        all_exp = expGoal + expColl


        diff_G0_last = abs(float(net.model(exG0.s0)[0][1] - AG))
        diff_C0_last = abs(float(net.model(exC0.s0)[0][1] - AC))

        def print_exp(header, exp, head):
            if head:
                print(header + ":")
                print("v0\tv1\tr\td\ta")
            for ex in exp:
                v0 = float(net.model(ex.s0)[0][2])
                v1 = float(net.model(ex.s1)[0][2])
                r = ex.r1
                d = r + settings.learning.gamma * v1 - v0
                a = float(net.model(ex.s1)[0][1])
                print("%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (v0, v1, r, d, a))

        # Look for overall improvement in 5 iterations
        # of a few steps each
        for j in range(0, 5):
            #print("Split round training %d" % (j+1))
            for i in range(0, 20):
                for ex in all_exp:
                    net.train_sample(ex)

            print_exp("Vals",all_exp, True)

            # Can't be as sure with training with both sets
            # So only test final results
            diff_G0 = abs(float(net.model(exG0.s0)[0][1] - AG))
            diff_C0 = abs(float(net.model(exC0.s0)[0][1] - AC))

        
        # Final States should have clear learning
        # Unsure of proper check for goal state
        self.assertGreater(diff_C0, diff_C0_last)



if __name__ == '__main__':
    test = TestNetworkBasics()
    test.test_smoke()