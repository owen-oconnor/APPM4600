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

def newton(f, fd, p0, tol, Nmax):
  """
  Newton iteration.
  
  Inputs:
    f, fd - function and its derivative
    x0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    p_star - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1)
  p[0] = p0
  for it in range(Nmax):
      p1 = p0 - f(p0)/fd(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          p_star = p1
          info = 0
          return [p, p_star, info, it]
      p0 = p1
  p_star = p1
  info = 1
  return [p, p_star, info, it]


'''Question 1'''

x_hat = 5 # try 5 meters as max depth
t_days = 60
t_seconds = t_days * 24 * 60 * 60 # convert 60 days into seconds
tol = 1e-13
alpha = 0.138e-6

def temp(x):
    T = erf(x / (2*np.sqrt(alpha*t_seconds))) - 15/35
    return T

def temp_prime(x):
    T_prime = np.exp(-(x / (2*np.sqrt(alpha*t_seconds)))**2) / (np.sqrt(alpha*t_seconds*np.pi))
    return T_prime

x = np.linspace(0, x_hat, 1000)
temps = temp(x)

plt.plot(x, temps)
plt.axhline(0, color='black')
plt.xlabel("Depth (m)")
plt.ylabel("Temp (C)")
plt.title("Temperature vs Depth after 60 Days")
plt.show()

root_bi, err = bisection(temp, 0, x_hat, tol=tol)
print(f'The approximate depth (root) using bisection is {root_bi} meters')

root_newt = newton(temp, temp_prime, p0=0.01, tol=tol, Nmax=200)[1]
print(f'The approximate depth (root) using Newton is {root_newt} meters')

root_newt2 = newton(temp, temp_prime, p0=5, tol=tol, Nmax=200)[1] # try initial guess of 5 meters
print(f'The approx depth (root) using Newton with initial guess of 5 meters is {root_newt2}')


'''Question 4'''
def f4(x):
    return np.exp(3*x) - 27*x**6 + 27*x**4*np.exp(x) - 9*x**2*np.exp(2*x)


'''Question 5'''
def f5(x):
    return x**6 - x - 1