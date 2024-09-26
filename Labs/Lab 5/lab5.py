import numpy as np

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