import numpy as np
import matplotlib.pyplot as plt
import time
from math import floor
from numba import jit


def verlet_step(r_old, r, dt, L, rc ):
    F, virial = LJ_Forces(r, L, rc)
    r_new = 2 * r + F * dt**2 - r_old
    return r_new, virial