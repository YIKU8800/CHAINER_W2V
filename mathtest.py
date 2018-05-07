import numpy as np


arra = np.array([0., 0., 1, 0., 0.])

arrb = np.array([0, 0, 1, 1, 0])

print(np.count_nonzero(arra == arrb))