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
from rl_car_simulator.controllers import Controller, HardCodedController, Controllers, ControllerTypes
from rl_car_simulator.experience_preprocessor import ExperiencePreprocessor
from rl_car_simulator.experience_engine import ExperienceEngine

# Values get overwritten by keyboard settings
AG = -0.2
AC = 0.3

class TestNetworkIntegration(unittest.TestCase):

    coll_exp_1 = None
    goal_exp_2 = None
    network_1 = None

    @classmethod
    def setUpClass(cls):
        global AG
        global AC
        settings = Settings()
        AG = -settings.keyboard.angle
        AC = settings.keyboard.angle
        settings.walls.walls = settings.walls.outer_walls
        settings.learning.max_episode_length = 200
        settings.physics.physics_timestep = 0.2
        settings.physics.control_timestep = 0.2
        settings.learning.alpha = 1e-5
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
        settings.preprocessing.use_types = [ControllerTypes.hardcoded]
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
        car.set_name("test")
        car.set_type(ControllerTypes.hardcoded)
        world.all_cars = [car]
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))
        controller = HardCodedController(settings, 1.0, 0.0)
        controllers = Controllers(None, None, controller, [], None, [], [])
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
        assert(i > 5)

        experience_raw = preprocessor.experience_queue.pop(0)
        #print([ex.r1 for ex in experience_raw])
        assert(len(experience_raw) > 2)

        experience = preprocessor.preprocess_episode(experience_raw)
        return experience, net

    def generate_collision_processed_experience_turning(settings):
        settings.preprocessing.use_types = [ControllerTypes.hardcoded]
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
        car.set_name("test")
        car.set_type(ControllerTypes.hardcoded)
        world.all_cars = [car]
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))
        controller = HardCodedController(settings, 1.0, AC)
        controllers = Controllers(None, None, controller, [], None, [], [])
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
        assert(i > 5)

        experience_raw = preprocessor.experience_queue.pop(0)
        #print([ex.r1 for ex in experience_raw])
        assert(len(experience_raw) > 2)

        experience = preprocessor.preprocess_episode(experience_raw)
        return experience, net

    def generate_goal_processed_experience(settings):
        settings.preprocessing.use_types = [ControllerTypes.hardcoded]
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
        car.set_name("test")
        car.set_type(ControllerTypes.hardcoded)
        world.all_cars = [car]
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))
        controller = HardCodedController(settings, 1.0, 0.0)
        controllers = Controllers(None, None, controller, [], None, [], [])
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
        assert(i > 5)

        experience_raw = preprocessor.experience_queue.pop(0)
        assert(len(experience_raw) > 2)

        experience = preprocessor.preprocess_episode(experience_raw)
        return experience, net

    def generate_goal_processed_experience_turning(settings):
        settings.preprocessing.use_types = [ControllerTypes.hardcoded]
        world = WorldCreation(settings).get()
        preprocessor = ExperiencePreprocessor(settings)
        experience = ExperienceEngine(settings, world, preprocessor)
        physics = PhysicsEngine(settings, world, experience)
        cs = CarState()
        cs.x = 20.0
        cs.y = 10.0
        cs.h = 0.0
        car = Car(settings, cs)
        car.goal = [22.0, 9.0]
        car.reached_goal = False
        world.keyboard_cars = []
        world.network_cars = []
        world.hardcoded_cars = [car]
        car.set_name("test")
        car.set_type(ControllerTypes.hardcoded)
        world.all_cars = [car]
        s0 = physics.get_car_state(car)
        net = Network(settings, len(s0))
        controller = HardCodedController(settings, 1.0, AG)
        controllers = Controllers(None, None, controller, [], None, [], [])
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
        assert(i > 3)

        experience_raw = preprocessor.experience_queue.pop(0)
        assert(len(experience_raw) > 2)

        experience = preprocessor.preprocess_episode(experience_raw)
        return experience, net

    def get_network_copy(self):
        net = Network(self.network_1.settings, self.network_1.N)
        #net.compile()
        return net






    def test_collision_learning(self):
        experience = copy.deepcopy(self.coll_exp_1)
        net = self.get_network_copy()
        
        ex0 = experience[0]
        exM = experience[int(len(experience)/2)]
        exF = experience[-1]
        try:
            v0_last = float(net.model(ex0.s0).value)
        except AssertionError as e:
            print(e)
            vM_last = float(net.model(exM.s0).value)
        vM_last = float(net.model(exM.s0).value)
        vF_last = float(net.model(exF.s0).value)

        v0 = float(net.model(ex0.s0).value)
        vM = float(net.model(exM.s0).value)
        vF = float(net.model(exF.s0).value)

        for i in range(0, 5):
            data, original = net.build_epoch_targets(experience)
            net.fit_model(data)
            
            v0 = float(net.model(ex0.s0).value)
            vM = float(net.model(exM.s0).value)
            vF = float(net.model(exF.s0).value)

            self.assertLess(v0, v0_last)
            self.assertLess(vM, vM_last)
            self.assertLess(vF, vF_last)

            v0_last = v0
            vM_last = vM
            vF_last = vF

    def test_goal_learning(self):
        experience = copy.deepcopy(self.goal_exp_1)
        net = self.get_network_copy()
        
        ex0 = experience[0]
        exM = experience[int(len(experience)/2)]
        exF = experience[-1]

        v0_last = float(net.model(ex0.s0).value)
        vM_last = float(net.model(exM.s0).value)
        vF_last = float(net.model(exF.s0).value)

        for i in range(0, 3):
            data, original = net.build_epoch_targets(experience)
            net.fit_model(data * 1000)
            
            v0 = float(net.model(ex0.s0).value)
            vM = float(net.model(exM.s0).value)
            vF = float(net.model(exF.s0).value)

            # Note: can have some variance in results
            #  with number of samples used, so may see
            #  from other changes
            self.assertGreater(v0, v0_last)
            self.assertGreater(vM, vM_last)
            self.assertGreater(vF, vF_last)

            v0_last = v0
            vM_last = vM
            vF_last = vF


    def test_split_experience_learning(self):
        settings = Settings()
        settings.learning.alpha = 1e-3
        settings.learning.gamma = 0.9
        expGoal = copy.deepcopy(self.goal_exp_1)
        expColl = copy.deepcopy(self.coll_exp_1)
        net = self.get_network_copy()
        
        exG0 = expGoal[0]
        exGF = expGoal[-1]
        exC0 = expColl[0]
        exCF = expColl[-1]

        expColl.reverse()
        all_exp = expGoal + expColl

        vG0_last = float(net.model(exG0.s0).value)
        vGF_last = float(net.model(exGF.s0).value)
        vC0_last = float(net.model(exC0.s0).value)
        vCF_last = float(net.model(exCF.s0).value)


        # Look for overall improvement in 5 iterations
        # of a few steps each
        for j in range(0, 5):
            data, original = net.build_epoch_targets(all_exp)
            net.fit_model(data * 200)

            # Can't be as sure with training with both sets
            # So only test final results
            vG0 = float(net.model(exG0.s0).value)
            vGF = float(net.model(exGF.s0).value)
            vC0 = float(net.model(exC0.s0).value)
            vCF = float(net.model(exCF.s0).value)

        # Final States should have clear learning
        self.assertGreater(vGF, vGF_last)
        self.assertLess(vCF, vCF_last)

        self.assertLess(vC0, vC0_last)
        self.assertGreater(vG0, vG0_last)


    def test_split_experience_actor_turning(self):
        settings = Settings()
        settings.learning.alpha = 1e-3
        settings.learning.gamma = 0.9
        settings.statistics.sigma = 0.5
        expGoal = copy.deepcopy(self.goal_exp_2)
        expColl = copy.deepcopy(self.coll_exp_2)
        net = self.get_network_copy()
        
        exG0 = expGoal[0]
        exGF = expGoal[-1]
        exC0 = expColl[0]
        exCF = expColl[-1]

        expColl.reverse()
        all_exp = expGoal + expColl

        #for ex in expColl:
        #    print(ex.G)
        
        ind_G0_last = 0
        ind_C0_last = 2
        prob_G0_last = net.model(exG0.s0).angle.action[ind_G0_last]
        prob_C0_last = net.model(exC0.s0).angle.action[ind_C0_last]

        #print("*****")

        # Look for overall improvement in 5 iterations
        # of a few steps each
        for j in range(0, 5):
            #print("Split round training %d" % (j+1))
            #print("a first: %f" % float(net.model(exG0.s0).angle ))
            data, original = net.build_epoch_targets(all_exp)
            net.fit_model(data * 200)

            #print_exp("Vals",all_exp, True)

            # Can't be as sure with training with both sets
            # So only test final results
            '''
            aG = float(net.model(exG0.s0).angle.action)
            aC = float(net.model(exC0.s0).angle.action)
            vG0 = float(net.model(exG0.s0).value)
            vC0 = float(net.model(exC0.s0).value)
            vG1 = float(net.model(exG0.s1).value)
            vC1 = float(net.model(exC0.s1).value)
            advG = float(net.predict_advantage(exG0))
            advC = float(net.predict_advantage(exC0))
            diff_G0 = abs(aG - AG)
            diff_C0 = abs(aC - AC)
            print("(G,C): pred(%f, %f)->v0(%f,%f)->v1(%f,%f)->adv(%f,%f)->act(%f,%f)" % (aG,aC,vG0,vC0,vG1,vC1,advG,advC,AG,AC))
            '''
            print("Goal(%s)[%d], Coll(%s)[%d]" % (str(net.model(exG0.s0).angle.action),
                                                      ind_G0_last,
                                                  str(net.model(exC0.s0).angle.action),
                                                      ind_C0_last))
            prob_G0 = net.model(exG0.s0).angle.action[ind_G0_last]
            prob_C0 = net.model(exC0.s0).angle.action[ind_C0_last]
            

        
        # Final States should have clear learning
        # Unsure of proper check for goal state
        self.assertGreater(prob_G0, prob_G0_last)
        self.assertLess(prob_C0, prob_C0_last)



if __name__ == '__main__':
    test = TestNetworkBasics()
    test.test_smoke()