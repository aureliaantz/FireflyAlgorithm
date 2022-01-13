import numpy as np


def objfRastrigin(x):       #2D Shifted Rastrigin's function
    sum = 0
    for i in range (len(x)):
        sum+=np.sum(np.power(x[i],2)-10*np.cos(2*np.pi*x[i]))+10

    return sum


def sphere(x):
    return np.sum(np.power(x, 2))


def objfRosenbrock(x):      #2D Shifted Rosenbrock's function
    sum = 0
    for z in x:
        sum += np.power((np.power(z,2)-(z+1)),2)*100+np.power(x-1,2)
    return sum


def objfGriewank(x):        #2D Shifted Griewank's function
    sum = np.sum(np.power(x,2))
    product = 1
    for i in range(len(x)):
        product *= np.cos(x[i] / np.sqrt(i + 1))
    return 1 + sum / 4000 - product


