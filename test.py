import physics
import plot
import dynamics
import numpy as np
import matplotlib.pyplot as plt


def test_LennardJonesForce():
    print(physics.LennardJonesForce(100000, 1000), "large distance")
    print(physics.LennardJonesForce(0.001, 1000), "minimum distance")


def test_LennardJonesPotential():
    print(physics.LennardJonesPotential(2 ** (1 / 6), 1000), "need to be -1")


def test_partical_drawing():
    r_new, Temperature, Pressure, Energy = dynamics.T0_config(10 ** (-4), 5, 2)
    plot.partical_drawing(r_new)


def test_Temperature():
    r_new, Temperature1, Pressure, Energy = dynamics.T0_config(10 ** (-4), 5, 2)
    r_new, Temperature10, Pressure, Energy = dynamics.T0_config(10 ** (-4), 5, 2)
    r_new, Temperature100, Pressure, Energy = dynamics.T0_config(10 ** (-4), 5, 2)
    r_new, Temperature1000, Pressure, Energy = dynamics.T0_config(10 ** (-4), 5, 2)
    r_new, Temperature10000, Pressure, Energy = dynamics.T0_config(10 ** (-4), 5, 2)
    r_new, Temperature100000, Pressure, Energy = dynamics.T0_config(10 ** (-4), 5, 2)

    x = np.array([1, 10, 100, 1000, 10000, 100000])
    y = np.array([Temperature1, Temperature10, Temperature100, Temperature1000, Temperature10000, Temperature100000])
    plt.plot(x, y)
    plt.title("Visualisation - Temperature")
    plt.xlabel("N")
    plt.ylabel("Temperature")
    plt.grid()
    plt.show()
