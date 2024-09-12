import numpy as np

def bisection(f, a, b, tol):
    fa = f(a)
    fb = f(b)

    if fa*fb > 0:
        err = 1
        root = "no root found"
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

f1 = lambda x: (x**2)*(x-1)
f2 = lambda x: (x-1)*(x-3)*(x-5)
f3 = lambda x: ((x-1)**2)*(x-3)
f4 = lambda x: np.sin(x)

tolerance = 10**(-5)

f5 = lambda x: x*(1+(7-x**5)/(x**2))**3
f6 = lambda x: x - (x**5 - 7)/(x**2)
f7 = lambda x: x - (x**5 - 7)/(5*x**4)
f8 = lambda x: x - (x**5 - 7)/(12)

# exercise 1a: use f1, a=0.5, b=2

root, error = bisection(f1, 0.5, 2, tolerance)
print(f'The approximate root beteen x=0.5, x=2 is {root}')

# exercise 1b: use f1, a=-1, b=0.5
root, error = bisection(f1, -1, 0.5, tolerance)
print(f'The approximate root beteen x=-1, x=0.5 is {root}')
#Unable to find double root at x = 0 because there is no sign change

# exercise 1c: use f1, a=-1, b=2
root, error = bisection(f1, -1, 2, tolerance)
print(f'The approximate root beteen x=-1, x=2 is {root}')
# only able to find one of the two roots in this interval