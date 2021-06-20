import numpy as np

class Wall:
    def __init__(self, data):
        p0 = data[0]
        p1 = data[1]

        self.x1 = np.array([[p0[0]],[p0[1]]]).reshape(-1)
        self.x2 = np.array([[p1[0]],[p1[1]]]).reshape(-1)

        inFrom1 = self.x2 - self.x1
        self.inFrom1 = (inFrom1 / np.linalg.norm(inFrom1, 2)).reshape(-1)
        self.inFrom2 = -inFrom1

        self.norm = np.array([[self.inFrom1[1]],[-self.inFrom1[0]]])
    
    def point_is_bounded(self, x):
        x = x.reshape(-1)
        
        v1 = x - self.x1
        a = np.dot(v1, self.inFrom1)

        v2 = x - self.x2
        b = np.dot(v2, self.inFrom2)
        return a > 0 and b > 0
    
    def get_point_side(self, x):
        x = x.reshape(-1)
        v = x - self.x1
        a = np.dot(v, self.norm)
        if a >= 0:
            return 1
        else:
            return -1