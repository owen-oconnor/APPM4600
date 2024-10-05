import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv

'''Question 1'''

def f1(x, y):
    return 3*x**2 - y**2

def g1(x, y):
    return 3*x*y**2 - x*3 - 1

def jacob(x, y):
    jacobian = np.array([[6*x, -2*y],
                            [3*y**2 - 3*x**2, 6*x*y]])
    return jacobian

def iterate1(f, g, v0, tol, Nmax):
    x0 = v0[0]
    y0 = v0[1]
    f0 = f(x0, y0)
    g0 = g(x0, y0)

    for i in range(Nmax):
        x1 = x0 - (1/6 * f0) - (1/18 * g0)
        y1 = y0 - 1/6 * g0

        v0 = np.array([x0, y0])
        v1 = np.array([x1, y1])

        if norm(v1-v0) < tol:
            return v1
        
        x0 = x1
        y0 = y1

    v = np.array([x1, y1])

    return [v, i]

def newton(v0, tol, Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    x0 = v0[0]
    y0 = v0[1]

    for its in range(Nmax):
       J = jacob(x0, y0)
       Jinv = inv(J)
       F = np.array([f1(x0, y0), g1(x0, y0)])
       
       v1 = v0 - Jinv.dot(F)
       
       if (norm(v1-v0) < tol):
           v = v1
           ier =0
           return[v, ier, its]
           
       v0 = v1
    
    v = v1
    ier = 1
    return[v, ier, its]

v0 = np.array([1, 
                1]) 
tol = 1e-10
Nmax = 1000

'''1a'''
scheme = iterate1(f1, g1, v0, tol, Nmax)
print(f'The iteration scheme converges on {scheme[0]}')


'''1b'''
newtons = newton(v0, tol, Nmax)
print(f'The newton method converges on {newtons[0]}')

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
