import numpy as np
import matplotlib as plt
import physics
from random import random
import plot


def T0_config(dt, N, L, rc):
    Energy = []
    Temp = []
    Pressure = []
    Counter = []
    r_old = L * np.random.rand(N, 2)
    r = r_old
    r_new, virial = physics.verlet_step(r_old, r, dt, L, rc)
    r_new = np.remainder(r_new, L)
    for i in range(100):
        r_old = r_new
        r = r_new
        r_new, virial = physics.verlet_step(r_old, r, dt, L, rc)
        r_new = np.remainder(r_new, L)
    i = 0
    while not convergence_condition(r_old, r, r_new, dt, L, rc, N):
        i += 1
        r_old = r_new
        r = r_new
        r_new, virial = physics.verlet_step(r_old, r, dt, L, rc)
        r_new = np.remainder(r_new, L)
        if not i % 1000:
            Counter.append(i)
            EK, EP, ET = physics.system_energy(r_old, r, r_new, dt, L, rc)
            Energy.append(ET)
            Temp.append(EK)
            Pressure.append(physics.pressure_virial(virial, EK, L))
    return r_new, Temp, Pressure, Energy, Counter



def convergence_condition(r_old, r, r_new, dt, L, rc, N):
    EK, EP, ET = physics.system_energy(r_old, r, r_new, dt, L, rc)
    return EK / N < 10**(-10)


if __name__ == '__main__':
    plot.particle_drawing(T0_config(10**(-4), 5, 10, 5)[0])










