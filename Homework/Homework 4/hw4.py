from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt

'''Define Methods'''
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

def newton(f, fd , x0, tol, Nmax):
  """
  Newton iteration.
  
  Inputs:
    f, fd - function and its derivative
    x0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1)
  p[0] = x0
  for it in range(Nmax):
      p1 = p0-f(p0)/fd(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]


'''Question 1'''

x_hat = 5 # try 5 meters as max depth
t_days = 60
t_seconds = t_days * 24 * 60 * 60 # convert 60 days into seconds
tol = 1e-13
alpha = 0.138e-6

def temp(x):
    T = 35*erf(x / (2*np.sqrt(alpha*t_seconds))) - 15
    return T

x = np.linspace(0, x_hat, 1000)
temps = temp(x)

plt.plot(x, temps)
plt.show()

root_bi, err = bisection(temp, 0, x_hat, tol=tol)
print(f'The approximate depth (root) using bisection is {root_bi} meters')

root_newt = newton(temp, temp, x0=0.01, tol=tol, Nmax=200)


'''Question 4'''
def f4(x):
    return np.exp(3*x) - 27*x**6 + 27*x**4*np.exp(x) - 9*x**2*np.exp(2*x)


'''Question 5'''
def f5(x):
    return x**6 - x - 1