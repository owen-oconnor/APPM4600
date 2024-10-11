import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

'''Define methods'''

def broyden():
    pass

def secant(f, x0, tol, Nmax):
    """
    Applies secant method to approximate root of a given function

    Args:
        f: the function we want to find the root of
        x0: the initial starting point for the method
        tol: the tolerance of the root
        Nmax: an upper bound on the number of iterations
    """

    p = [x0, x1]
    for it in range(Nmax):
        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        p.append(x_new)
        if abs(x_new - x1) < tol:
            return [p, x_new, it]
        x0, x1 = x1, x_new
    return [p, x_new, it]

def newton(f, j, x0, tol, Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = j(x0)
       Jinv = inv(J)
       F = f(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return [xstar, ier, its]

def steepest_descent(f, j, v0, tol, Nmax, alpha=0.1):
    for its in range(Nmax):
        F = f(v0)
        J = j(v0)
        v1 = v0 - 2*alpha*J.T @ F

        if norm(v1 - v0) < tol:
            vstar = v1
            return [vstar, its]
        
        v0 = v1

    return [v1, its]


'''Question 1'''

v = np.array([1, 
             1])

def f1(v):
    x = v[0]
    y = v[1]
    return np.array([x**2 + y**2 - 4,
                     np.exp(x) + y - 1])

def J1():
    x = v[0]
    y = v[1]

    J = np.zeros((2,2))

    J[0, 0] = 2*x
    J[0, 1] = 2*y

    J[1, 0] = np.exp(x)
    J[1, 1] = 1

    return J

'''sol_broyden = broyden()
sol_secant = secant()

print(f'The approximate solution using Broyden method is {sol_broyden[0]} in {sol_broyden[1]} iterations')
print(f'The approximate solution using the Secant method is {sol_secant[0]} in {sol_secant[1]} iterations')'''


'''Question 2'''

def f2(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return np.array([x + np.cos(x*y*z) - 1,
                     (1-x)**(1/4) + y + 0.05*z**2 - 0.15*z -1,
                     -x**2 - 0.1*y**2 + 0.01*y + z - 1])

def J2(v):
    x = v[0]
    y = v[1]
    z = v[2]

    J = np.zeros((3,3))

    J[0, 0] = 1 - y*z*np.sin(x*y*z)
    J[0, 1] = - x*z*np.sin(x*y*z)
    J[0, 2] = - x*y*np.sin(x*y*z)

    J[1, 0] = -0.25 * (1-x)**(-3/4)
    J[1, 1] = 1
    J[1, 2] = 0.1*z - 0.15

    J[2, 0] = -2*x
    J[2, 1] = -0.2*y + 0.01
    J[2, 2] = 1

    return J

tol = 5e-2
v2 = np.array([0.5,
              0.5,
              0.5])

sol_newt = newton(f2, J2, v2, tol, Nmax=500)
sol_steep = steepest_descent(f2, J2, v2, tol, Nmax=500)
print(f2(sol_newt[0]))
print(f2(sol_steep[0]))

print(f'The approximate solution using Newtons method is {sol_newt[0]} in {sol_newt[1] + 1} iterations')
print(f'The approximate solution using the Steepest Descent method is {sol_steep[0]} in {sol_steep[1] + 1} iterations')