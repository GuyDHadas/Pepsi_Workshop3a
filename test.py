import physics
def test_LennardJonesForce():
    print(physics.LennardJonesForce(100000, 5), "large distance")
    print(physics.LennardJonesForce(2 ** (-1/6), 5), "minimum distance")


test_LennardJonesForce()