import numpy as np

def bisection(f, a, b, tol):
    fa = f(a)
    fb = f(b)

    if fa*fb > 0:
        err = 1
        return err
    
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

f1 = lambda x: (x**2)*(x-1)
f2 = lambda x: (x-1)*(x-3)*(x-5)
f3 = lambda x: ((x-1)**2)*(x-3)
f4 = lambda x: np.sin(x)

tolerance = 10**(-5)

f5 = lambda x: x*(1+(7-x**5)/(x**2))**3
f6 = lambda x: x - (x**5 - 7)/(x**2)
f7 = lambda x: x - (x**5 - 7)/(5*x**4)
f8 = lambda x: x - (x**5 - 7)/(12)