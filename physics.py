import numpy as np
import matplotlib.pyplot as plt
import time
from math import floor
from numba import jit


def running_time(func, x):
    t_start = time.time()
    func(x)
    t_end = time.time()
    return t_end - t_start


# r_vec is a np. array of D elements (D is the number of dimensions ).
# it is the vector which points from particle 1 to particle 2 (= r_ij =ri -rj)
# this function returns the Lennard - Jones potential between the two particles
def LennardJonesPotential(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0.
    rc6 = rc * rc * rc * rc * rc * rc
    rc12 = rc6 * rc6
    r6 = r * r * r * r * r * r
    r12 = r6 * r6
    return 4 / r12 - 4 / r6 - 4 / rc12 + 4 / rc6


# same as previous method but returns the force between the two particles
# this is the gradient of the previous method
def LennardJonesForceFast(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return 0. * r_vec
    y = 1 / r
    y2 = y * y
    y4 = y2 * y2
    y8 = y4 * y4
    return (48 * y8 * y8 * r * r - 24 * y8) * r_vec
    # calculate the gradient of " LennardJonesPotential # this method calculates the total force on each particle


# r is a 2D array where r[i ,:] is a vector with length D ( dimensions )
# which represents the position of the i-th particle
# (in 2D case r[i ,0] is the x coordinate and r[i ,1] is the y coordinate of the i-th # this function returns a numpy array F of the same dimesnsions as r
# where F[i ,:] is a vector which represents the force that acts on the i-th particle
# this function also returns the virial
def LJ_Forces(r, L, rc):
    F = np.zeros_like(r)
    virial = 0
    N = r.shape[0]  # number of particles
    # loop on all pairs of particles i, j
    for i in range(1, N):
        for j in range(i):
            r_ij = r[i, :] - r[j, :]
            r_ij = r_ij - L * np.rint(r_ij / L)  # see class on boundary f_ij = LennardJonesForce (r_ij , rc)
            F[i, :] += f_ij
            F[j, :] -= f_ij  # third law of newton
            virial += np.dot(f_ij, r_ij)  # see class on virial theorem
    return F, virial




def system_energy(r_old, r, r_new, dt, L, rc):
    """

    :param r_old:
    :param r:
    :param r_new:
    :param dt:
    :param L:
    :param rc:
    :return:
    """
    T = 0
    for i in range(len(r)):
        r_tmp = (r_new[i] - r_old[i])/dt
        T = T + 0.125 * np.dot(r_tmp, r_tmp)
    V = 0
    for i in range(len(r)):
        for j in range(i +1, len(r)):
            print(r[i] - r[j])
            V = V + LennardJonesPotential(r[i]-r[j], rc)
    return T + V




if __name__ == '__main__':
    x = np.array([1,2])
    system_energy(np.array([x,x]),np.array([x,x]),np.array([x,x]),1,1,1)


def verlet_step(r_old, r, dt, L, rc ):
    F, virial = LJ_Forces(r, L, rc)
    r_new = 2 * r + F * dt**2 - r_old
    return r_new, virial
