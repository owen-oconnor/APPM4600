import numpy as np
import math
from numpy.linalg import norm
from numpy.linalg import inv

'''Before Lab'''

def forward_diff(f, s, h):
    f_prime = (f(s+h) - f(s)) / h
    return f_prime

def centered_diff(f, s, h):
    f_prime = (f(s + h) - f(s)) / 2*h
    return f_prime

f = lambda x: np.cos(x)
s = np.pi / 2
h_values = 0.01 * 2.0 ** (-np.arange(0, 10))

forward_diffs = [forward_diff(f, s, h) for h in h_values]
centered_diffs = [centered_diff(f, s, h) for h in h_values]
print(forward_diffs)
print(centered_diffs)


def Newton(f, j, x0, tol, Nmax):

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
           
def LazyNewton(f, j, x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = j(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = f(x0)
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return [xstar, ier, its]   

def slacker_newton(f1, j, x0, tol, Nmax):

    ''' Slacker Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = j(x0)
    Jinv = inv(J)
    update_count = 0
    for its in range(Nmax):

       f = f1(x0)
       x1 = x0 - Jinv.dot(f)
       
       if norm(-Jinv.dot(f)) > 1e-2: # update Jacobian if step is too big
           J = j(x1)
           Jinv = inv(J)
           update_count += 1 # count how many times jacobian is updated  

       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return [xstar, ier, its, update_count]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return [xstar, ier, its]  


'''Exercises'''

# 3.2

def func(x):
    f = np.array([4*x[0]**2 + x[1]**2 - 4,
                   x[0] + x[1] - np.sin(x[0] - x[1])])
    return f

def jacob(x): # calculate jacobian for f
    j = np.array([[8*x[0], 2*x[1]],
                  [1 - np.cos(x[0] - x[1]), 1 + np.cos(x[0] - x[1])]])
    return j

x0 = np.array([1, 
               0])
tol = 1e-10


lazy = LazyNewton(func, jacob, x0, tol, 500)
root = lazy[0]
iters = lazy[2]
print(f'The approx root with lazy newton is {root} in {iters} iterations')


slacker = slacker_newton(func, jacob, x0, tol, 500)
root = slacker[0]
iters = slacker[2]
updates = slacker[3]
print(f'The approx root with slacker newton is {root} in {iters} iterations with {updates} updates to the Jacobian')

# 3.3