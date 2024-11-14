import numpy as np
import matplotlib.pyplot as plt
from adaptive_quad import *
from scipy.integrate import quad

a = 0.1
b = 2.
n = 5
tol = 1e-3

f = lambda x: np.sin(1/x)

I_trap, x, nsplit_trap = adaptive_quad(a, b, f, tol, n, method=eval_composite_trap)
I_simps, x, nsplit_simps = adaptive_quad(a, b, f, tol, n, method=eval_composite_simpsons)
I_gauss, x, nsplit_gauss = adaptive_quad(a, b, f, tol, n, method=eval_gauss_quad)
I_real, _ = quad(f, a, b)
print(f'The approximate integral with composite trapezoidal is {I_trap} in {nsplit_trap} intervals')
print(f'The approximate integral with composite simpsons is {I_simps} in {nsplit_simps} intervals')
print(f'The approximate integral with gaussian quad is {I_gauss} in {nsplit_gauss} intervals')
print(f'The approximate integral using scipy.integrate.quad is {I_real}')


