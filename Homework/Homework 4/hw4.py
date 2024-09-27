from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from ...methods.root_finders import bisection

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
  p = [p0]
  for it in range(Nmax):
      p1 = p0 - f(p0)/fd(p0)
      p.append(p1)
      if (abs(p1-p0) < tol):
          p_star = p1
          info = 0
          return [p, p_star, info, it]
      p0 = p1
  p_star = p1
  info = 1
  return [p, p_star, info, it]

def modified_newt(f, df, m, p0, tol, Nmax):
    p = [p0]
    for it in range(Nmax):
        p1 = p0 - (m*f(p0))/df(p0)
        p.append(p1)
        if (abs(p1-p0) < tol):
            p_star = p1
            info = 0
            return [p, p_star, info, it]
    p0 = p1
    p_star = p1
    info = 1
    return [p, p_star, info, it]

def secant(f, x0, x1, tol, Nmax):
    p = [x0, x1]
    for it in range(Nmax):
        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        p.append(x_new)
        if abs(x_new - x1) < tol:
            return [p, x_new, it]
        x0, x1 = x1, x_new
    return [p, x_new, it]

def calculate_slope(log_prev_errors, log_errors):
    slope, intercept = np.polyfit(log_prev_errors, log_errors, 1)
    return slope

def order_of_convergence(x_values, root, method_name):
    errors = [abs(x - root) for x in x_values if abs(x-root) != 0]
    log_errors = np.log(errors[1:])
    log_prev_errors = np.log(errors[:-1])
    
    # Calculate the slope
    slope = calculate_slope(log_prev_errors, log_errors)
    
    # Plot the log-log graph
    plt.figure()
    plt.plot(log_prev_errors, log_errors, '-', label=f'{method_name}')
    plt.xlabel(r'$\log(|x_k - \alpha|)$')
    plt.ylabel(r'$\log(|x_{k+1} - \alpha|)$')
    plt.title(f'Order of Convergence - {method_name}')
    plt.show()

    return slope


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
f4 = lambda x: np.exp(3*x) - 27*x**6 + 27*x**4*np.exp(x) - 9*x**2*np.exp(2*x)
df4 = lambda x: 3*np.exp(3*x) - (6*27*(x**5)) + 4*27*(x**3)*np.exp(x) + 27*x**4*np.exp(x) - 2*9*x*np.exp(2*x) - 2*9*x**2*np.exp(2*x) 
m = 1 # multiplicity for modified method

newt = newton(f4, df4, p0=4, tol=1e-8, Nmax=500)
values_newt = newt[0]
root_newt = newt[1]

'''mod_newt = modified_newt(f4, df4, m, p0=4, tol=1e-8, Nmax=500)
values_mod_newt = mod_newt[0]
root_mod_newt = mod_newt[1]
print(values_mod_newt, root_mod_newt)'''

order_newt = order_of_convergence(values_newt, root_newt, 'Newton')
#order_mod_newt = order_of_convergence(values_mod_newt, root_mod_newt, 'Modified Newt')
print(f'The order of convergence with the newton method (slope of log-log graph) is {order_newt}')
#print(f'The order of convergence with the secant method (slope of log-log graph) is {order_mod_newt}')

'''Question 5'''
f5 = lambda x: x**6 - x - 1
df5 = lambda x: 6*(x**5) - 1

def errors(values, root):
    errs = [abs(v - root) for v in values]
    return errs

x0 = 2
x1 = 1
tol = 1e-13

newt = newton(f5, df5, x0, tol, Nmax=500)
values_newt = newt[0]
root_newt = newt[1]
iters = newt[3]
errs_newt = errors(values_newt, root_newt)
print(errs_newt)


sec = secant(f5, x0, x1, tol, Nmax=500)
values_sec = sec[0]
root_sec = sec[1]
errs_sec = errors(values_sec, root_sec)
print(errs_sec)

order_newt = order_of_convergence(values_newt, root_newt, 'Newton')
order_sec = order_of_convergence(values_sec, root_sec, 'Secant')
print(f'The order of convergence with the newton method (slope of log-log graph) is {order_newt}')
print(f'The order of convergence with the secant method (slope of log-log graph) is {order_sec}')