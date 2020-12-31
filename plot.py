import matplotlib.pyplot as plt
import numpy as np
import physics




def visualization(r_c):
    """

    :param r_c:
    :param r:
    :return:
    """
    b = np.linspace(0,2.1,10000)
    x = r_c * (0.1 **b)
    y = np.array([physics.LennardJonesPotential(np.array(a), r_c) for a in x])
    plt.plot(x, y)
    plt.xlabel("distance")
    plt.ylabel("Potential")
    plt.xscale('log')
    plt.grid()
    plt.show()




if __name__ == '__main__':

    visualization(100)