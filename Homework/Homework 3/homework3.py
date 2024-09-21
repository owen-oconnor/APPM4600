import numpy as np
import matplotlib.pyplot as plt


''' Define Methods '''

def bisection(f, a, b, tol):
    fa = f(a)
    fb = f(b)

    if fa*fb > 0:
        err = 1
        root = "no root found"
        return root, err
    
    if fa == 0:
        root = fa
        err = 0
        return root, err
    elif fb == 0:
        root = fb
        err = 0
        return root, err
    
    d = 0.5*(a+b)
    while abs(d-a) > tol:
        fd = f(d)

        if f(d)*f(a) > 0:
            a = d
            fa = fd
        else:
            b = d

        d = 0.5*(a+b)
        fd = f(d)

    root = d
    err = 0

    return root, err

def fixed_point(f, x0, tol, Nmax):
    count = 0

    while count < Nmax:
       count += 1
       x1 = f(x0)
       if abs(x1 - x0) < tol:
          xstar = x1
          err = 0
          return xstar, err
       x0 = x1

    xstar = x1
    err = 1
    return xstar, err

########################################################

'''Questions'''
#1c

# Question 2

f_binom = lambda x: (x-5)**9 
f_poly = lambda x: x**9 - 45*(x**8) + 900*(x**7) - 10500*(x**6) + 78750*(x**5) - 393750*(x**4) + 1312500*(x**3) - 2812500*(x**2) + 3515625*x - 1953125 # full expansion of binomial (x-5)**9
a = 4.82
b = 5.2
tol = 1e-4

# 2a
root, error = bisection(f_binom, a, b, tol)
print(f'The approximate root is {root}')

# 2b
root, error = bisection(f_poly, a, b, tol)
print(f'The approximate root is {root}')


###########################

# Question 5

def f(x):
    return -np.sin(2*x) + (5*x)/4 - (3/4)

x = np.linspace(0, 3*np.pi, 200)
y = x - 4*np.sin(2*x) - 3
plt.plot(x, y)
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

def f(x):
    return -np.sin(2*x) + (5*x)/4 - (3/4)