import time
import numpy as np
import math

class Utility:
    def __init__(self):
        pass

    def now(self):
        return time.time()

    def rot(self, r):
        c = np.cos(r)
        s = np.sin(r)
        rot = np.array([[c,-s], [s,c]])
        rot = rot.reshape((2,2))
        return rot

    def normal_density(self, x, u, sig):
        fac = 1.0 / (sig * math.sqrt(2*math.pi))
        power = -1/2 * math.pow((x-u)/sig, 2)
        return fac * math.exp(power)
    
    def normal_density_derivative(self, x, u, sig):
        return self.normal_density(x,u,sig) * (x - u) / (sig * sig)
    
    def normal_int_width(self, sig):
        # Guarantee that density * width < 1.0
        return math.sqrt(2 * math.pi) * sig * 0.01
    
    def normal_int_prob(self, x, u, sig):
        density = self.normal_density(x,u,sig)
        width = self.normal_int_width(sig)
        return density * width