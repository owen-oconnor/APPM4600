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

def newton(f, df, p0, tol, Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
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
  p[0] = p0
  for it in range(Nmax):
      p1 = p0 - f(p0)/df(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]


def hybrid(f, df, ddf, a, b, tol, Nmax):
    """
    Hybrid Iteration ethod: 
    
    Inputs:
    f: function you want to find the root of
    df: derivative of f
    ddf: second derivative of f
    a: left endpoint for bisection
    b: right endpoint for bisection
    tol: tolerance for newton's method
    Nmax: max number of iterations

    Outputs:
    p: array of all root approximations at each iteration of newton
    pstar: approximate root to tolerance provided
    it: number of iterations
    info: success/failure information (0 for success, 1 for fail)
    
    """
    fa = f(a)
    fb = f(b)

    if fa*fb > 0:
        err = 1
        root = "no root in specified interval"
        return root, err
    
    if fa == 0:
        root = a
        err = 0
        return root, err
    elif fb == 0:
        root = b
        err = 0
        return root, err
    
    m = 0.5*(a+b)
    basin_check = (ddf(m) * f(m) / df(m)**2) - 1 # check if midpoint is in basin of convergence for Newton's method
    while basin_check > 1:
        fm = f(m)

        if f(m)*f(a) > 0:
            a = m
            fa = fm
        else:
            b = m

        m = 0.5*(a+b)
        fm = f(m)
        basin_check = (ddf(m) * f(m) / df(m)**2) - 1

    p0 = m

    # now apply Newton's method using midpoint that we know is in basin of convergence

    p = np.zeros(Nmax+1)
    p[0] = p0
    for it in range(Nmax):
      p1 = p0 - f(p0)/df(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          p_star = p1
          info = 0
          return [p, p_star, info, it]
      p0 = p1
    p_star = p1
    info = 1
    return [p, p_star, info, it]
