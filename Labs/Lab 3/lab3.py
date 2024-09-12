import numpy as np

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
exercise = '1a'
a = 0.5
b = 2
root, error = bisection(f1, a, b, tolerance)
print(f'Exercise {exercise}: The approximate root beteen x={a}, x={b} is {root}')

# exercise 1b: use f1, a=-1, b=0.5
exercise = '1b'
a = -1
b = 0.5
root, error = bisection(f1, a, b, tolerance)
print(f'Exercise {exercise}: The approximate root beteen x={a}, x={b} is {root}')
#Unable to find double root at x = 0 because there is no sign change

# exercise 1c: use f1, a=-1, b=2
exercise = '1c'
a = -1
b = 2
root, error = bisection(f1, a, b, tolerance)
print(f'Exercise {exercise}: The approximate root beteen x={a}, x={b} is {root}')
# only able to find one of the two roots in this interval

# exercise 2a: use f2, a = 0, b = 2.4 and same tolerance as above
exercise = '2a'
a = 0
b = 2.4
root, error = bisection(f2, a, b, tolerance)
print(f'Exercise {exercise}: The approximate root beteen x={a}, x={b} is {root}')


# exercise 2b: use f3, a = 0, b = 2 and same tolerance as above
exercise = '2b'
a = 0
b = 2
root, error = bisection(f3, a, b, tolerance)
print(f'Exercise {exercise}: The approximate root beteen x={a}, x={b} is {root}')

# exercise 2c: use f4, a = 0, b = 0.1 and same tolerance as above
exercise = '2c'
a = 0
b = 0.1
root, error = bisection(f4, a, b, tolerance)
print(f'Exercise {exercise}: The approximate root beteen x={a}, x={b} is {root}')


##################################################
'''Fixed Point Eval'''
x0 = 1
tolerance = 10**(-10)
Nmax = 100
test_x = 7**(1/5)

# exercise 3a: use function f5 
exercise = '3a'
check = f5(test_x)
if abs(check - test_x) < tolerance:
    print(f'Ex {exercise}: {test_x} is a verified fixed point of function')
try:
    fp, error = fixed_point(f5, x0, tolerance, Nmax)
    print(f'Exercise {exercise}: The approximate fixed point w/ inital guess of {x0} is {fp}')
except OverflowError:
    print(f'Ex {exercise}: Algorithm diverges, unable to approx fixed point')

# exercise 3b: use function f6, other parameters unchanged
exercise = '3b'
if abs(check - test_x) < tolerance:
    print(f'Ex {exercise}: {test_x} is a verified fixed point of function')
try:
    fp, error = fixed_point(f6, x0, tolerance, Nmax)
    print(f'Exercise {exercise}: The approximate fixed point w/ inital guess of {x0} is {fp}')
except OverflowError:
    print(f'Ex {exercise}: Algorithm diverges, unable to approx fixed point')

#exercise 3c: use function f7
exercise = '3c'
if abs(check - test_x) < tolerance:
    print(f'Ex {exercise}: {test_x} is a verified fixed point of function')
try:
    fp, error = fixed_point(f7, x0, tolerance, Nmax)
    print(f'Exercise {exercise}: The approximate fixed point w/ inital guess of {x0} is {fp}')
except OverflowError:
    print(f'Ex {exercise}: Algorithm diverges, unable to approx fixed point')

# exercise 3d: use function f8
exercise = '3d'
if abs(check - test_x) < tolerance:
    print(f'Ex {exercise}: {test_x} is a verified fixed point of function')
try:
    fp, error = fixed_point(f8, x0, tolerance, Nmax)
    print(f'Exercise {exercise}: The approximate fixed point w/ inital guess of {x0} is {fp}')
except OverflowError:
    print("Algorithm diverges, unable to approx fixed point")