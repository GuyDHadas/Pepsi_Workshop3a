import numpy as np
import matplotlib.pyplot as plt
import time
from math import floor
from numba import jit



def running_time3(func, x1, x2, x3):
    t_start = time.time()
    func(x1, x2, x3)
    t_end = time.time()
    return t_end - t_start


# r is a 2D array where r[i ,:] is a vector with length D ( dimensions )
# which represents the position of the i-th particle
# (in 2D case r[i ,0] is the x coordinate and r[i ,1] is the y coordinate of the i-th particle)
# this function returns a numpy array F of the same dimesnsions as r
# where F[i ,:] is a vector which represents the force that acts on the i-th particle
# this function also returns the virial
def LennardJonesForce(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0. * r_vec
    return 4 * (12 / r ** 14 - 6 / r ** 8) * r_vec


def LJ_Forces(r, L, rc):
    F = np.zeros_like(r)
    virial = 0
    N = r.shape[0]  # number of particles
    # loop on all pairs of particles i, j
    for i in range(1, N):
        for j in range(i):
            r_ij = r[i, :] - r[j, :]
            r_ij = r_ij - L * np.rint(r_ij / L)  # see class on boundary
            f_ij: float = LennardJonesForce(r_ij, rc)
            F[i, :] += f_ij
            F[j, :] -= f_ij  # third law of newton
            virial += np.dot(f_ij, r_ij)  # see class on virial theorem
    return F, virial


def LJ_Forces2(r, L, rc):
    F = np.zeros_like(r)
    r_diff = np.zeros(r.shape, r.shape)
    virial = 0
    N = r.shape[0]  # number of particles
    # loop on all pairs of particles i, j
    for i in range(1, N):
        for j in range(i):
            r_diff[i][j] = r[i, :] - r[j, :]
    r_diff = r_diff - L * np.rint(r_diff / L)  # see class on boundary
    f_diff = LennardJonesForce(r_diff, rc)
    for i in range(N):
        F[i, :] += np.sum(f_diff[i][j])
    virial = np.dot(f_diff, r_diff)  # see class on virial theorem
    return F, virial


N_array = [floor(10 ** i) for i in np.linspace(2, 3, 2)]
times_python = []
times_np = []

for n in N_array:
    x = np.random.rand(n)
    times_python.append(running_time3(LJ_Forces, np.random.rand(n, 2), 1.0, 0.3))
    times_np.append(running_time3(LJ_Forces2, np.random.rand(n, 2), 1.0, 0.3))

plt.plot(N_array, times_python, label='given function - f')
plt.plot(N_array, times_np, label='my improvement')
plt.title("Code Accelerating - f")
plt.xlabel("$N$")
plt.ylabel("time [s]")
plt.xscale('log')
plt.grid()
plt.legend()

plt.show()

speedup = [times_python[i] / times_np[i] for i in range(len(times_np))]
plt.plot(N_array, speedup)
plt.title("Code Accelerating - f")
plt.xlabel("$N$")
plt.ylabel("speedup")
plt.xscale('log')
plt.grid()

plt.show()
