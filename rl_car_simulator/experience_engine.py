import math
import numpy as np

from .car import CarStepExperience

class ExperienceEngine:
    def __init__(self, settings, world, preprocessor):
        self.settings = settings
        self.world = world
        self.preprocessor = preprocessor

    def sample_rewards(self):
        for car in self.world.all_cars:
            if car.step_experience is not None:
                r = self.calculate_reward(car)
                car.step_experience.set_r1(r)

    def calculate_reward(self, car):
        r = -1.0
        if car.collided:
            r -= 75.0
        if car.reached_goal:
            r += 75.0
        return r

    def sample_end_states(self):
        for car in self.world.all_cars:
            if car.step_experience is not None:
                car.step_experience.set_s1(car.sensed_state)

    def sample_start_states(self):
        for car in self.world.all_cars:
            if car.step_experience is not None:
                car.step_experience.set_s0(car.sensed_state)

    def sample_controls(self):
        for car in self.world.all_cars:
            if car.step_experience is not None:
                car.step_experience.set_a0(car.controls)

    def new_experience_step(self):
        for car in self.world.all_cars:
            if car.step_experience is not None:
                car.episode_steps.append(car.step_experience)
            car.step_experience = CarStepExperience()

    def handle_episode_ends(self):
        for car in self.world.all_cars:
            if car.reached_goal:
                self.preprocessor.new_experience(car.episode_steps)
                car.episode_steps = []
                print("Reached Goal")
            elif car.collided:
                self.preprocessor.new_experience(car.episode_steps)
                car.episode_steps = []
