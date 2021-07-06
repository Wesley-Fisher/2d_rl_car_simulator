import time
import numpy as np

class Utility:
    def __init__(self):
        pass

    def now(self):
        return time.time()

    def rot(self, r):
        c = np.cos(r)
        s = np.sin(r)
        return np.array([[c,-s], [s,c]])