import config

if config.backend == "gpu":
    import cupy as xp
else:
    import numpy as xp

class Array:
    """Each array"""

    def __init__(self, r_a, element_distance, row_elements, column_elements):
        self.row_elements = row_elements
        self.column_elements = column_elements
        self.uni_distance = element_distance
        self.elements = row_elements * column_elements
        self.r_prime = xp.zeros((3, self.elements))

        # place all microphone elements at the right position
        element_index = 0
        for j in range(row_elements):
            for i in range(column_elements):
                self.r_prime[0, element_index] = i * self.uni_distance + r_a[0]
                self.r_prime[1, element_index] = j * self.uni_distance + r_a[1]
                element_index += 1

        # center matrix in origin (0,0)
        self.r_prime[0, :] = self.r_prime[0, :] - \
            self.row_elements*self.uni_distance/2 + self.uni_distance/2
        self.r_prime[1, :] = self.r_prime[1, :] - \
            self.column_elements*self.uni_distance/2 + self.uni_distance/2
