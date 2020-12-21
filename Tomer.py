import time
import numpy as np


def running_time(func, x, y):
    t_start = time.time()
    for i in range(1000000):
        func(x, y)
    t_end = time.time()
    return t_end - t_start


def LennardJonesPotential2(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0.
    rc6 = rc * rc * rc * rc * rc * rc
    rc12 = rc6 * rc6
    r6 = r * r * r * r * r * r
    r12 = r6 * r6
    return 4 / r12 - 4 / r6 - 4 / rc12 + 4 / rc6


def LennardJonesPotential(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0.
    VLJ_rc = 4 * (1 / rc ** 12 - 1 / rc ** 6)
    return 4 * (1 / r ** 12 - 1 / r ** 6) - VLJ_rc


print(running_time(LennardJonesPotential2, 10000000, 5))
print(running_time(LennardJonesPotential, 10000000, 5))


print(LennardJonesPotential2(100000000, 5))
print(LennardJonesPotential(100000000, 5))
