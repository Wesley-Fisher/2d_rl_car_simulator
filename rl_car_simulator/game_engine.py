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
from .controllers import KeyboardController, NetworkController, Controllers
from .experience_engine import ExperienceEngine
from .experience_preprocessor import ExperiencePreprocessor

class GameEngine:
    def __init__(self, settings):
        self.util = Utility()
        self.settings = settings
        self.world = WorldCreation(self.settings).get()

        self.experience_preprocessor = ExperiencePreprocessor(self.settings)
        self.experience = ExperienceEngine(self.settings, self.world, self.experience_preprocessor)

        self.graphics = Graphics(self.settings, self.world)
        self.physics = PhysicsEngine(self.settings, self.world, self.experience)
        
        
        null_state = self.physics.get_car_state(Car(self.settings, CarState()))
        self.network = Network(self.settings, len(null_state))
        null_out = self.network.get(null_state)

        keyboard = KeyboardController(self.settings)
        network = NetworkController(self.settings, self.network)
        self.controllers = Controllers(keyboard, network, None)
        self.physics.set_controllers(self.controllers)

        
        
        self.running = True
        self.time_start = self.util.now()

        self.kill_check_thread = threading.Thread(target=self.check_end_fn)
        self.graphics_thread = threading.Thread(target=self.graphics_fn)
        self.physics_thread = threading.Thread(target=self.physics_fn)
        self.preprocess_thread = threading.Thread(target=self.preprocess_fn)
        self.training_thread = threading.Thread(target=self.training_fn)

    def run(self):
        self.kill_check_thread.start()
        self.graphics_thread.start()
        self.physics_thread.start()
        self.preprocess_thread.start()
        self.training_thread.start()

        while self.running:
            time.sleep(1.0)

    def check_end_fn(self):
        while self.running:
            time.sleep(1.0)
            if self.util.now() - self.time_start > self.settings.application.time_limit:
                self.running = False

    def graphics_fn(self):
        while self.running:
            self.graphics.show_current_world()
            self.graphics.sleep()
        self.graphics.close()

    def physics_fn(self):
        t_last_controls = self.util.now() - self.settings.physics.control_timestep*2.0
        while self.running:
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
        while self.running:
            time.sleep(0.1)
            self.network.add_new_experience()

            if len(self.network.training_experience) < self.settings.memory.min_train_size:
                continue

            stats, training_results = self.network.train_epoch()
            num_rem = self.network.remove_samples(training_results)
            print("Training Stats:")
            print("Num Samples: %d" % stats.num_samples)
            print("Removed: %d" % num_rem)
            self.network.save_state()
