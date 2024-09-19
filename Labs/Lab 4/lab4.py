import numpy as np
import math

def fixed_point(f, x0, tol, Nmax):
    count = 0
    values = []
    while count < Nmax:
       count += 1
       x1 = f(x0)
       values.append(x1)
       if abs(x1 - x0) < tol:
          err = 0
          return np.array(values), err
       x0 = x1

    err = 1
    return np.array(values), err

def order_of_convergence(seq, p_true):
    errors = np.abs(seq - p_true)  
    orders = []
    
    for n in range(1, len(errors) - 1):
        alpha = np.log(errors[n+1]/errors[n]) / np.log(errors[n]/errors[n-1])
        orders.append(alpha) 
    
    return orders


g = lambda x: math.sqrt(10 / (x+4))
sequence, err = fixed_point(g, 1.5, 1e-10, 200)
print(sequence)
ord = order_of_convergence(sequence, p_true=1.3652300134140976)
print(ord)