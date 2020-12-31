import physics
import plot
import dynamics


def test_LennardJonesForce():
    print(physics.LennardJonesForce(100000, 1000), "large distance")
    print(physics.LennardJonesForce(0.001, 1000), "minimum distance")


def test_LennardJonesPotential():
    print(physics.LennardJonesPotential(2 ** (1 / 6), 1000), "need to be -1")


def test_partical_drawing():
    r_new = dynamics.T0_config(10 ** (-4), 5, 2)
    plot.partical_drawing(r_new)


test_LennardJonesForce()
