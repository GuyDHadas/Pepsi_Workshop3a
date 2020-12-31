import physics
def test_LennardJonesForce():
    print(physics.LennardJonesForce(100000, 1000), "large distance")
    print(physics.LennardJonesForce(0.001, 1000), "minimum distance")


def test_LennardJonesPotential():
    print(physics.LennardJonesPotential(2 ** (1/6), 1000), "need to be -1")




test_LennardJonesForce()