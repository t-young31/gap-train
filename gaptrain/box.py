import numpy as np


class Box:

    def random_point(self):
        """Get a random point inside the box"""
        return np.array([np.random.uniform(0.0, k) for k in self.size])

    def __init__(self, size):

        assert len(size) == 3
        self.size = [float(k) for k in size]
