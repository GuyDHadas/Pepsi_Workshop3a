import matplotlib.pyplot as plt
import numpy as np
import physics


def visualization(r_c):
    """

    :param r_c:
    :param r:
    :return:
    """
    x = np.linspace(0.85, 10, 1000)
    y = np.array([physics.LennardJonesPotential(np.array(a), r_c) for a in x])
    plt.plot(x, y)
    plt.title("Visualisation - Potential")
    plt.xlabel(r'distance $[\sigma]$')
    plt.ylabel(r'Potential $[\epsilon]$')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    visualization(100)
