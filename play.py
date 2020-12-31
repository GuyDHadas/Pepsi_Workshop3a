import numpy as np
import matplotlib.pyplot as plt
import time
from math import floor
from numba import jit


def running_time(func, *args):
    t_start = time.time()
    for i in range(10000):
        func(*args)
    t_end = time.time()
    return t_end - t_start


def running_time2(func, *args):
    t_start = time.time()
    func(*args)
    t_end = time.time()
    return t_end - t_start

# r_vec is a np . array of D elements ( D is the number of dimensions ).
# it is the vector which points from particle 1 to particle 2 (= r_ij = ri - rj )
# this function returns the Lennard - Jones potential between the two particles
def LennardJonesPotential(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0.
    VLJ_rc = 4 * (1 / rc**12 - 1 / rc**6)
    return 4 * (1 / r**12 - 1 / r**6) - VLJ_rc


@jit
def LennardJonesPotential2(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0.
    rc2 = rc * rc
    rc6 = rc2 * rc2 * rc2
    rc12 = rc6 * rc6
    r2 = r * r
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    return 4 / r12 - 4 / r6 - 4 / rc12 + 4 / rc6


print(LennardJonesPotential(np.linspace(0, 10, 10000), 3))
print(LennardJonesPotential2(np.linspace(0, 10, 10000), 3))

N_array = [floor(10**i) for i in np.linspace(3, 5, 10)]
times_given = []
times_ours = []

for n in N_array:
    x = np.random.rand(n)
    times_given.append(running_time(LennardJonesPotential, x, 3))
    times_ours.append(running_time(LennardJonesPotential2, x, 3))

plt.plot(N_array, times_given, label='given function')
plt.plot(N_array, times_ours, label='our improvement')
plt.title("Code Accelerating - LennardJonesPotential")
plt.xlabel("$N$")
plt.ylabel("time [s]")
plt.xscale('log')
plt.grid()
plt.legend()

plt.show()

speedup = [times_given[i]/times_ours[i] for i in range(len(times_ours))]
plt.plot(N_array, speedup)
plt.title("Code Accelerating -  LennardJonesPotential")
plt.xlabel("$N$")
plt.ylabel("speedup")
plt.xscale('log')
plt.grid()

plt.show()


# same as previous method but returns the force between the two particles
# this is the gradient of the previous method
def LennardJonesForce(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0. * r_vec
    return 4 * (12 / r**14 - 6 / r**8) * r_vec  # calculate the gradient of "LennardJonesPotential" to get this formula


@jit
def LennardJonesForce2(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0. * r_vec
    y = 1 / r
    y2 = y * y
    y4 = y2 * y2
    y6 = y4 * y2
    y8 = y4 * y4
    y14 = y6 * y8
    return (48 * y14 - 24 * y8) * r_vec


print(LennardJonesForce(np.linspace(0, 10, 100000), 3))
print(LennardJonesForce2(np.linspace(0, 10, 100000), 3))

N_array = [floor(10**i) for i in np.linspace(3, 5, 10)]
times_given = []
times_ours = []

for n in N_array:
    x = np.random.rand(n)
    times_given.append(running_time(LennardJonesForce, x, 3))
    times_ours.append(running_time(LennardJonesForce2, x, 3))

plt.plot(N_array, times_given, label='given function')
plt.plot(N_array, times_ours, label='our improvement')
plt.title("Code Accelerating - LennardJonesForce")
plt.xlabel("$N$")
plt.ylabel("time [s]")
plt.xscale('log')
plt.grid()
plt.legend()

plt.show()

speedup = [times_given[i]/times_ours[i] for i in range(len(times_ours))]
plt.plot(N_array, speedup)
plt.title("Code Accelerating -  LennardJonesForce")
plt.xlabel("$N$")
plt.ylabel("speedup")
plt.xscale('log')
plt.grid()

plt.show()


# this method calculates the total force on each particle
# r is a 2 D array where r [i ,:] is a vector with length D ( dimensions )
# which represents the position of the i - th particle
# ( in 2 D case r [i ,0] is the x coordinate and r [i ,1] is the y coordinate of the i - th particle )
# this function returns a numpy array F of the same dimensions as r
# where F [i ,:] is a vector which represents the force that acts on the i - th particle
# this function also returns the virial
def LJ_Forces(r, L, rc):
    F = np.zeros_like(r)
    virial = 0
    N = r.shape[0]  # number of particles
    # loop on all pairs of particles i , j
    for i in range(1, N):
        for j in range(i):
            r_ij = r[i, :] - r[j, :]
            r_ij = r_ij - L * np.rint(r_ij / L)  # see class on boundary conditions
            f_ij = LennardJonesForce(r_ij, rc)
            F[i, :] += f_ij
            F[j, :] -= f_ij  # third law of newton
            virial += np.dot(f_ij, r_ij)  # see class on virial theorem
    return F, virial


@jit
def LJ_Forces2(r, L, rc):
    F = np.zeros_like(r)
    virial = 0
    N = r.shape[0]  # number of particles
    # loop on all pairs of particles i , j
    for i in range(1, N):
        for j in range(i):
            r_ij = r[i, :] - r[j, :]
            r_ij = r_ij - L * np.rint(r_ij / L)  # see class on boundary conditions
            f_ij = LennardJonesForce2(r_ij, rc)
            F[i, :] += f_ij
            F[j, :] -= f_ij  # third law of newton
            virial += np.dot(f_ij, r_ij)  # see class on virial theorem
    return F, virial


print(LJ_Forces(np.random.rand(1000, 2), 1, 3))
print(LJ_Forces2(np.random.rand(1000, 2), 1, 3))

N_array = [floor(10**i) for i in np.linspace(2, 3, 5)]
times_given = []
times_ours = []

for n in N_array:
    x = np.random.rand(n, 2)
    times_given.append(running_time2(LJ_Forces, x, 1, 0.5))
    times_ours.append(running_time2(LJ_Forces2, x, 1, 0.5))

plt.plot(N_array, times_given, label='given function')
plt.plot(N_array, times_ours, label='our improvement')
plt.title("Code Accelerating - LJ_Forces")
plt.xlabel("$N$")
plt.ylabel("time [s]")
plt.xscale('log')
plt.grid()
plt.legend()

plt.show()

speedup = [times_given[i]/times_ours[i] for i in range(len(times_ours))]
plt.plot(N_array, speedup)
plt.title("Code Accelerating -  LJ_Forces")
plt.xlabel("$N$")
plt.ylabel("speedup")
plt.xscale('log')
plt.grid()

plt.show()

