import numpy as np
import matplotlib.pyplot as plt

coeffs = [0, 1, 0, -1/6, 0, 1/120, 0] # taylor series coefficients for sin(x)
order_arrangements = [(3, 3), (2, 4), (4, 2)]
f = lambda x: np.sin(x)

x = np.linspace(0, 5, 1000)
f_exact = f(x)
maclaurin_f = np.polyval(coeffs[::-1], x)