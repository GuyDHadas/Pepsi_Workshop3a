import numpy as np
import matplotlib as plt
import physics
from random import random


def T0_config(dt, N, L, rc):

    r_old = L * np.random.rand(N, 2)
    r = r_old
    r_new, virial = physics.verlet_step(r_old, r, dt, L, rc)
    r_new = np.remainder(r_new, L)
    for i in range(100):
        r_old = r_new
        r = r_new
        r_new, virial = physics.verlet_step(r_old, r, dt, L, rc)
        r_new = np.remainder(r_new, L)
    while not convergence_condition(r_old, r, r_new, dt, L, rc, N):
        r_old = r_new
        r = r_new
        r_new, virial = physics.verlet_step(r_old, r, dt, L, rc)
        r_new = np.remainder(r_new, L)
    return r_new


def convergence_condition(r_old, r, r_new, dt, L, rc, N):
    EK, EP, ET = physics.system_energy(r_old, r, r_new, dt, L, rc)
    print(EK / N)
    return EK / N < 10**(-9)


if __name__ == '__main__':
    print(T0_config(10**(-4), 5, 10, 5))








