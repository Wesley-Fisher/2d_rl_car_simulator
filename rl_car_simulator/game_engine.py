#!/usr/bin/env python3

from os import stat
import time
import threading

from .settings import Settings
from .graphics import Graphics
from .utilities import Utility
from .world_creation import WorldCreation
from .physics_engine import PhysicsEngine
from .network import Network
from .car import Car, CarState
from .controllers import ExplorationController, FeedbackController, KeyboardController, NetworkController, RandomController, Controllers
from .experience_engine import ExperienceEngine
from .experience_preprocessor import ExperiencePreprocessor
from .reporting import Reporting

class GameEngine:
    def __init__(self, settings):
        self.util = Utility()
        self.settings = settings

        self.train_only = False

        self.world = WorldCreation(self.settings).get()

        self.reporting = Reporting(self.settings)
        self.experience_preprocessor = ExperiencePreprocessor(self.settings, self.reporting)
        self.experience = ExperienceEngine(self.settings, self.world, self.experience_preprocessor)

        self.graphics = Graphics(self.settings, self.world)
        self.physics = PhysicsEngine(self.settings, self.world, self.experience)
        
        
        null_state = self.physics.get_car_state(Car(self.settings, CarState()))
        self.network = Network(self.settings, len(null_state))
        null_out = self.network.get(null_state)

        keyboard = KeyboardController(self.settings)
        network = NetworkController(self.settings, self.network)
        random = [RandomController(self.settings) for car in self.world.random_cars]
        feedback = FeedbackController(self.settings)
        network_exploration = [ExplorationController(self.settings, network) for car in self.world.network_exploration_cars]
        feedback_exploration = [ExplorationController(self.settings, feedback) for car in self.world.feedback_exploration_cars]
        self.controllers = Controllers(keyboard, network, None, random, feedback, network_exploration, feedback_exploration)
        self.physics.set_controllers(self.controllers)

        
        
        self.running = True
        self.time_start = self.util.now()

        self.kill_check_thread = threading.Thread(target=self.check_end_fn)
        self.graphics_thread = threading.Thread(target=self.graphics_fn)
        self.physics_thread = threading.Thread(target=self.physics_fn)
        self.preprocess_thread = threading.Thread(target=self.preprocess_fn)
        self.training_thread = threading.Thread(target=self.training_fn)
        self.reporting_thread = threading.Thread(target=self.reporting_fn)

    def run(self):
        self.kill_check_thread.start()
        self.graphics_thread.start()
        self.physics_thread.start()
        self.preprocess_thread.start()
        self.training_thread.start()
        self.reporting_thread.start()

        while self.running:
            time.sleep(1.0)

    def check_end_fn(self):
        while self.running:
            time.sleep(1.0)
            if self.util.now() - self.time_start > self.settings.application.time_limit:
                self.running = False

    def wait_for_training(self):
        waited = self.train_only
        while self.running and self.train_only:
            time.sleep(1.0)
        return waited

    def graphics_fn(self):
        while self.running:
            self.wait_for_training()

            self.graphics.show_current_world()
            self.graphics.sleep()
        self.graphics.close()

    def physics_fn(self):
        t_last_controls = self.util.now() - self.settings.physics.control_timestep*2.0
        while self.running:
            self.wait_for_training()

            t_loop_start = self.util.now()

            if self.util.now() - t_last_controls > self.settings.physics.control_timestep:
                self.network.freeze()
                self.physics.full_control_sensor_step()

                t_last_controls = self.util.now()
            
            self.physics.full_physics_termination_step()

            dt = self.util.now() - t_loop_start
            t_sleep = max(self.settings.physics.physics_timestep - dt, 0.0)
            time.sleep(t_sleep)
    

    def preprocess_fn(self):
        while self.running:
            time.sleep(1.0)

            while len(self.experience_preprocessor.experience_queue) > 0:
                exp = self.experience_preprocessor.experience_queue.pop(0)
                exp = self.experience_preprocessor.preprocess_episode(exp)
                self.network.add_experience(exp)
    
    def training_fn(self):
        try:
            self.network.load_state()
        except OSError as e:
            print("Could not load network and/or memory from file")

        while self.running:
            time.sleep(0.1)
            self.network.add_new_experience()

            l_exp = len(self.network.training_experience)
            if l_exp < self.settings.memory.min_train_size:
                continue

            if self.settings.memory.size_train_only > 0 and l_exp > self.settings.memory.size_train_only:
                print("Only training network until experience discarded; Halting world")
                self.train_only = True

            if self.settings.memory.size_resume_world > 0 and l_exp < self.settings.memory.size_resume_world:
                print("Resuming world")
                self.train_only = False

            print("Starting epoch on %d samples" % l_exp)

            sample_results, epoch_results = self.network.train_epoch()
            num_rem = self.network.remove_samples(sample_results)
            print("Training Stats:")
            print("Removed: %d" % num_rem)
            print("Avg Critic Step: %f" % epoch_results.avg_c_step)
            print("Avg Actor F Step: %f" % epoch_results.avg_af_step)
            print("Avg Actor A Step: %f" % epoch_results.avg_aa_step)
            self.network.save_state()

    def reporting_fn(self):
        last_car_performance_time = self.util.now()

        while self.running:
            waited = self.wait_for_training()
            if waited:
                last_car_performance_time = self.util.now()

            time.sleep(1.0)
            now = self.util.now()

            if now - last_car_performance_time > self.settings.reporting.car_performance_report_interval:
                self.reporting.report_car_performance()
                last_car_performance_time = now
