import numpy as np
import matplotlib.pyplot as plt

def energy_func_value(z: np.array, order: int):
    '''
        z: np.array (2,)
        z1 = y-axis, z2 = x-axis
    '''

    w1 = lambda z: np.sin(2*np.pi*z[0]/4)

    if order == 1:
        result = ((np.linalg.norm(z)-2)/0.4)**2/2-np.log(np.exp(-((z[0]-2)/0.6)**2)+np.exp(-((z[0]+2)/0.6)**2))
    elif order == 2:
        result = ((z[1]-w1(z))/0.4)**2/2

    result = np.exp(-result)

    return result

def energy_func(order: int):
    height, width = 256, 256

    array = np.zeros((height, width))

    for i, w in enumerate(np.linspace(start=-4, stop=4, num=height)):
        for j, h in enumerate(np.linspace(start=-4, stop=4, num=width)):
            array[i][j] = energy_func_value(np.array([h, w]), order=order)

    return array

plt.imshow(energy_func(1), cmap='jet')
plt.savefig('subtasks/01_energy_func/output.png', dpi=500)