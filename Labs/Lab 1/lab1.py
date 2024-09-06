import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 100)
y = np.arange(100)

print(f'The first 3 entries of x are: {x[0]}, {x[1]}, and {x[2]}')
print(f'The first 3 entries of y are: {y[0]}, {y[1]}, and {y[2]}')

w = 10**(-np.linspace(1,10,10))
x = np.linspace(1, len(w))
s = 3*w