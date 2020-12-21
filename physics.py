# r_vec is a np. array of D elements (D is the number of dimensions ).
#it is the vector which points from particle 1 to particle 2 (= r_ij =ri -rj)
# this function returns the Lennard - Jones potential between the two particles
def LennardJonesPotential (r_vec , rc ):
    r = np. linalg . norm ( r_vec ) # calculate norm (= | r_ij |)
    if r > rc: return 0.
        VLJ_rc = 4 * (1/ rc **12 - 1/ rc **6)
    return 4 * (1/ r **12 - 1/r **6) - VLJ_rc
# same as previous method but returns the force between the two particles
# this is the gradient of the previous method
def LennardJonesForce (r_vec , rc ):
    r = np. linalg . norm ( r_vec ) # calculate norm (= | r_ij |)
    if r > rc:
        return 0.* r_vec
    return 4 * (12/ r **14 - 6/r **8) * r_vec # calculate the gradient of " LennardJonesPotential # this method calculates the total force on each particle
# r is a 2D array where r[i ,:] is a vector with length D ( dimensions )
# which represents the position of the i-th particle
#(in 2D case r[i ,0] is the x coordinate and r[i ,1] is the y coordinate of the i-th # this function returns a numpy array F of the same dimesnsions as r
# where F[i ,:] is a vector which represents the force that acts on the i-th particle
# this function also returns the virial
def LJ_Forces (r, L, rc ):
    F = np. zeros_like (r)
    virial = 0
    N = r. shape [0] # number of particles
    # loop on all pairs of particles i, j
    for i in range (1, N):
        for j in range (i):
            r_ij = r[i, :] - r[j, :]
            r_ij = r_ij - L * np. rint ( r_ij / L) # see class on boundary f_ij = LennardJonesForce (r_ij , rc)
            F[i, :] += f_ij
            F[j, :] -= f_ij # third law of newton
            virial += np. dot (f_ij , r_ij ) # see class on virial theorem
    return F, virial