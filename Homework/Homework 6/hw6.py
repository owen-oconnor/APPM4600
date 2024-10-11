import numpy as np

'''Define methods'''

def broyden():
    pass

def secant(tol, Nmax):
    return

def newton(tol, Nmax):
    pass

def steepest_descent():
    return 0


'''Question 1'''

v = np.array(1, 
             1)

def f1(x, y):
    return np.array([x**2 + y**2 - 4,
                     np.exp(x) + y - 1])

def J1():
    jacobian = 0
    return jacobian


'''Question 2'''

def f2(x, y, z):
    return np.array([x + np.cos(x*y*z) - 1,
                     (1-x)**(1/4) + y + 0.05*z**2 - 0.15*z -1,
                     -x**2 - 0.1*y**2 + 0.01*y + z - 1])

def J2():
    jacobian = 0
    return jacobian

tol = 5e-2