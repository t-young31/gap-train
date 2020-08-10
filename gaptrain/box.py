import numpy as np


class Box:

    def random_grid_point(self, spacing):
        """Create a uniform grid of (x, y, z) points inside this box"""

        vector = []
        for length in self.size:
            value = np.random.choice(np.arange(0, length, spacing))
            vector.append(value)

        return np.array(vector)

    def random_point(self):
        """Get a random point inside the box"""
        return np.array([np.random.uniform(0.0, k) for k in self.size])

    def __init__(self, size):
        """Periodic cuboidal box"""

        assert len(size) == 3
        self.size = np.array([float(k) for k in size])
