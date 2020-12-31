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



def partical_drawing(r):
    x = np.array([a[0] for a in r])
    y = np.array([a[1] for a in r])
    plt.plot(x,y,'o', color = 'black')
    plt.show()


if __name__ == '__main__':
    visualization(100)
    r = np.array([[1,1],[1.2,2],[0,3],[1.1,0.9]])
    partical_drawing(r)
