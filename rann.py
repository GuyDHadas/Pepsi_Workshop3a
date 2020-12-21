import numpy as np
import matplotlib.pyplot as plt
import time
from math import floor
from numba import jit



def LennardJonesForce(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0. * r_vec
    return 4 * (12 / r ** 14 - 6 / r ** 8) * r_vec

@jit
def LennardJonesForceFast(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0. * r_vec
    y = 1/r
    y2 = y * y
    y4 = y2 * y2
    y8 = 24 * y4 * y4
    return y8 * (2*y8*r*r - 1) * r_vec

def running_time2(func, x, y):
    t_start = time.time()
    func(x, y)
    t_end = time.time()
    return t_end - t_start

if __name__ == '__main__':
    x = np.array(range(10000000))
    LennardJonesForceFast(x, 5)
    print(running_time2(LennardJonesForce, x, 5))
    print(running_time2(LennardJonesForceFast, x, 5))