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


g = lambda x: math.sqrt(10 / (x+4))
sequence, err = fixed_point(g, 1.5, 1e-10, 200)
print(sequence)