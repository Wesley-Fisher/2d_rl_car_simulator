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
        r = -0.01
        if car.collided:
            r -= 1.0
        if car.reached_goal:
            r += 1.0
        else:
            if car.step_experience is not None and \
               car.step_experience.s0 is not None and \
               car.step_experience.s1 is not None:
                dx0 = car.step_experience.s0[0] - car.goal[0]
                dy0 = car.step_experience.s0[1] - car.goal[1]
                dist_0 = dx0*dx0 + dy0*dy0

                dx1 = car.step_experience.s1[0] - car.goal[0]
                dy1 = car.step_experience.s1[1] - car.goal[1]
                dist_1 = dx1*dx1 + dy1*dy1

                r = r + (dist_0 - dist_1) / 100.0

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

            l = self.settings.learning.max_episode_length
            if l > 0 and len(car.episode_steps) > l:
                self.preprocessor.new_experience(car.episode_steps)
                car.episode_steps = []
