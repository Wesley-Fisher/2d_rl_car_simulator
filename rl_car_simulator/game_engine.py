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
from .experience_prprocessor import ExperiencePreprocessor

class GameEngine:
    def __init__(self):
        self.util = Utility()
        self.settings = Settings()
        self.world = WorldCreation(self.settings).get()

        self.experience_preprocessor = ExperiencePreprocessor(self.settings)

        self.graphics = Graphics(self.settings, self.world)
        self.physics = PhysicsEngine(self.settings, self.world)
        self.experience = ExperienceEngine(self.settings, self.world, self.experience_preprocessor)
        
        null_state = self.physics.get_car_state(Car(self.settings, CarState()))
        self.network = Network(self.settings, len(null_state))
        null_out = self.network.get(null_state)

        keyboard = KeyboardController(self.settings)
        network = NetworkController(self.settings, self.network)
        self.controllers = Controllers(keyboard, network)
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
                self.physics.sensors_step()

                self.experience.sample_end_states()
                self.experience.sample_rewards()

                self.experience.handle_episode_ends()
                self.physics.handle_resets()

                self.experience.new_experience_step()

                self.network.freeze()
                self.physics.controls_step()

                self.experience.sample_start_states()
                self.experience.sample_controls()

                t_last_controls = self.util.now()
            
            self.physics.physics_time_step()
            self.physics.handle_goals()
            self.physics.handle_collisions()

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
            stats = self.network.train_epoch()
            print("Training Stats:")
            print("Num Samples: %d" % stats.num_samples)
            print("Removed: %d" % stats.num_removed)
            time.sleep(5.0)
