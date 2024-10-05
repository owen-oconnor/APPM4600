import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv

'''Question 1'''

def f1(x, y):
    return 3*x**2 - y**2

def g1(x, y):
    return 3*x*y**2 - x*3 - 1

def jacob(x, y):
    jacobian = np.array([6*x, -2*y],
                            [3*y**2 - 3*x**2, 6*x*y])
    return jacobian

def iteration(f, g, v0, tol, Nmax):
    x0 = v0[0]
    y0 = v0[1]
    f0 = f(x0, y0)
    g0 = g(x0, y0)
    matrix = np.array([[1/6, 1/18],
                        [0, 1/6]])
    funcs = np.array([f0,
                        g0])

    for i in range(Nmax):
        v1 = v0 - matrix*funcs
        f1 = f(v1[0])
        g1 = f(v1[1])
        funcs = np.array([f1, 
                          g1])

    return v1, i

def newton(x0, tol, Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = jacob(x0)
       Jinv = inv(J)
       F = f1(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar, ier, its]

v0 = np.array([1, 
                1]) 



'''Question 3b'''

def f(x, y, z):
    return x**2 + 4*y**2 + 4*z**2 - 16

def f_x(x, y, z):
    return 2 * x

def f_y(x, y, z):
    return 8 * y

def f_z(x, y, z):
    return 8 * z

def iterate3(x_n, y_n, z_n):
    func_value = f(x_n, y_n, z_n)
    fx = f_x(x_n, y_n, z_n)
    fy = f_y(x_n, y_n, z_n)
    fz = f_z(x_n, y_n, z_n)
    
    denominator = fx**2 + fy**2 + fz**2
    d = func_value / denominator

    x_next = x_n - d * fx
    y_next = y_n - d * fy
    z_next = z_n - d * fz
    
    return x_next, y_next, z_next

x_0, y_0, z_0 = 1, 1, 1
tol = 1e-10
Nmax = 1000

x_n, y_n, z_n = x_0, y_0, z_0
errors = []
for i in range(Nmax):
    x_next, y_next, z_next = iterate3(x_n, y_n, z_n)
    
    error = np.sqrt((x_next - x_n)**2 + (y_next - y_n)**2 + (z_next - z_n)**2)
    errors.append(error)
    if i > 1:
        p = np.log(errors[i] / errors[i-1]) / np.log(errors[i-1] / errors[i-2])
        print(f'The order of convergence is {p}')
    
    print(f"Iteration {i}: x = {x_next}, y = {y_next}, z = {z_next}, error = {error}")
    
    if error < tol:
        print(f"Converged to (x, y, z) = ({x_next}, {y_next}, {z_next}) after {i} iterations.")
        break
    
    # Update for the next iteration
    x_n, y_n, z_n = x_next, y_next, z_next
else:
    print("Did not converge within the maximum number of iterations.")
