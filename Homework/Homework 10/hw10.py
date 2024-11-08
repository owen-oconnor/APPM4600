import numpy as np
import matplotlib.pyplot as plt

coeffs = [0, 1, 0, -1/6, 0, 1/120, 0] # taylor series coefficients for sin(x) up to order 6
f = lambda x: np.sin(x)

x = np.linspace(0, 5, 1000)
f_exact = f(x)
f_maclaurin = np.polyval(coeffs[::-1], x)
maclaurin_err = np.abs(f_exact - f_maclaurin)

pade_33 = lambda x: (x + (6/120 - 1/6)*x**3) / (1 + 6*x**2/120)
f_pade33 = pade_33(x)
pade33_err = np.abs(f_exact - f_pade33)

pade_24 = lambda x: (x) / (1 + x/6 + (1/36 - 1/120)*x**4)
f_pade24 = pade_24(x)
pade24_err = np.abs(f_exact - f_pade24)

pade_42 = lambda x: (x + (6/120 - 1/6)*x**3) / (1 + 6*x**2/120) # identical to p33
f_pade42 = pade_42(x)
pade42_err = np.abs(f_exact - f_pade42)

plt.semilogy(x, maclaurin_err, label='Maclaurin Series Error')
plt.semilogy(x, pade33_err, label='Pade Approx Error (m=3, n=3)')
plt.semilogy(x, pade24_err, label='Pade Approx (m=2, n=4)')
plt.semilogy(x, pade42_err, label='Pade Approx Error (m=4, n=2)')

plt.legend()
plt.title('Errors of Maclaurin and Pade Approximations')
plt.show()