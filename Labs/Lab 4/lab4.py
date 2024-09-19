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
          return np.array(values), err, count
       x0 = x1

    err = 1
    return np.array(values), err

def order_of_convergence(seq, p_true):
    diff1 = np.abs(seq[1::] - p_true)
    diff2 = np.abs(seq[0:-1] - p_true)
    
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)
    alpha = fit[0] 
    
    return alpha

def aitkens(seq):
    return


g = lambda x: math.sqrt(10 / (x+4))
sequence, err, count = fixed_point(g, 1.5, 1e-10, 200)
print(sequence, count)
ord = order_of_convergence(sequence, p_true=1.3652300134140976)
print(f'the order of convergence is {ord}')