import numpy as np


class TDExperience:
    def __init__(self):
        self.s0 = None
        self.a_force = None
        self.a_angle = None
        self.pf = None
        self.pa = None
        self.G = None
        self.s1 = None
        self.step_in_ep = 0
        self.next_terminal = False
        self.num_uses = 0

class ExperiencePreprocessor:
    def __init__(self, settings, reporting=None):
        self.settings = settings
        self.reporting = reporting

        self.experience_queue = []

    def new_experience(self, exp, type, name):
        
        if self.reporting is not None:
            self.reporting.record_car_performance(name, self.get_total_reward(exp))

        if type in self.settings.preprocessing.use_types:
            self.experience_queue.append(exp)
            #print("%s in %s" % (type, str(self.settings.preprocessing.use_types)))
        else:
            #print("%s not in %s" % (type, str(self.settings.preprocessing.use_types)))
            pass

    def get_total_reward(self, exp):
        total = 0.0
        for ex in exp:
            total = total + ex.r1
        return total

    def preprocess_episode(self, exp):
        #print("Adding episode with %d steps" % len(exp))
        G = 0.0
        exp.reverse()
        N = len(exp)

        out = []
        i = 0
        skipped = False
        for ex in exp:
            G = self.settings.preprocessing.gamma * G + ex.r1

            td = TDExperience()
            td.s0 = ex.s0.reshape((1,8))
            td.a_force = ex.a0.force
            td.a_angle = ex.a0.steer
            td.pf = ex.pf
            td.pa = ex.pa
            td.r1 = ex.r1
            td.s1 = ex.s1.reshape((1,8))
            td.step_in_ep = N - i
            td.G = G
            td.next_terminal = ( i == 0)

            if i % self.settings.preprocessing.subsample:
                out.append(td)
                skipped = True
            else:
                skipped = False
            i = i + 1

        # Always keep last step (with big rewards)
        if skipped:
            out.append(td)

        return out

