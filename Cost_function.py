import numpy as np


def objfRastrigin(x):       #2D Shifted Rastrigin's function
    dim = len(x)
    return np.sum(np.power(x,2)-10*np.cos(2*np.pi*x))+10*dim


def sphere(x):
    dim = len(x)
    return np.sum(np.power(x, 2))*dim


def objfRosenbrock(x):      #2D Shifted Rosenbrock's function
    for z in x:
        sum += np.power((np.power(z,2)-(z+1)),2)*100+np.power(x-1,2)
        #return sum(100.0(x[1:]-x[:-1]2.0)2.0 + (1-x[:-1])**2.0)
    return sum


def objfGriewank(x):        #2D Shifted Griewank's function
    dim = len(x)
    sum = 0
    sum = np.sum(np.power(x,2))+dim
    product = 1
    for i in range(len(x)):
        product *= np.cos(x[i] / np.sqrt(i + 1))
    return 1 + sum / 4000 - product


