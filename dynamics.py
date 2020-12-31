import numpy as np
import matplotlib as plt
import physics
from random import random


def T0_config(dt, N, L, rc):
    x_initial_locations = []
    y_initial_locations = []
    z_initial_locations = []
    x_locations = []
    y_locations = []
    z_locations = []
    r_old = []
    r = []
    r_new = []

    for i in range(N):
        x_initial_locations.append(np.random.random(L))
        y_initial_locations.append(np.random.random(L))
        z_initial_locations.append(np.random.random(L))
        x_locations.append(np.random.random(L))
        y_locations.append(np.random.random(L))
        z_locations.append(np.random.random(L))
        r_old.append([x_initial_locations[i], y_initial_locations[i], z_initial_locations[i]])
        r.append([x_locations[i], y_locations[i], z_locations[i]])

    while True:
        for i in range(N):
            r_new.append(physics.verlet_step(r_old, r, dt, L, rc))
            r_new[i] = np.remainder(L, r_new[i])
            r_old = r_new
            r = r_new

        return r





