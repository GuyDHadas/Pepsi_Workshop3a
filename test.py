import physics
def test_LennardJonesForce():
    print(physics.LennardJonesForceFast(100000, 5), "large distance")
    print(physics.LennardJonesForceFast(2 ** (-1/6), 5), "minimum distance")


test_LennardJonesForce()