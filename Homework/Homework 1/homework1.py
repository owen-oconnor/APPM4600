import matplotlib.pyplot as plt
import numpy as np

def poly_coeffs(x):
    return x**9 - 18*x**8 + 144*x**7 -672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512

def poly_binom(x):
    return (x-2)**9

xs = np.arange(1.920, 2.081, 0.001)
ys_coeffs = [poly_coeffs(i) for i in xs]
ys_binom = [poly_binom(i) for i in xs]

plt.plot(xs, ys_coeffs)
plt.plot(xs, ys_binom)
plt.show()