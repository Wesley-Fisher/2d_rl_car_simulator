import numpy as np


class TDExperience:
    def __init__(self):
        self.s0 = None
        self.a_force = None
        self.a_angle = None
        self.G = None
        self.s1 = None
        self.step_in_ep = 0
        self.next_terminal = False

class ExperiencePreprocessor:
    def __init__(self, settings):
        self.settings = settings

        self.experience_queue = []

    def new_experience(self, exp):
        self.experience_queue.append(exp)

    def preprocess_episode(self, exp):
        #print("Adding episode with %d steps" % len(exp))
        G = 0.0
        exp.reverse()
        N = len(exp)

        out = []
        i = 0
        for ex in exp:
            G = self.settings.preprocessing.gamma * G + ex.r1

            td = TDExperience()
            td.s0 = ex.s0.reshape((1,8))
            td.a_force = ex.a0.force
            td.a_angle = ex.a0.steer
            td.r1 = ex.r1
            td.s1 = ex.s1.reshape((1,8))
            td.step_in_ep = N - i
            td.G = G
            td.next_terminal = ( i == 0)

            out.append(td)
            i = i + 1
        return out

