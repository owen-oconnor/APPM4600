import numpy as np

'''Define Methods'''

def bisection_modified(f, df, ddf, a, b, tol, Nmax):
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
    basin_check = (ddf(m) * f(m) / df**2) - 1 # check if midpoint is in basin of convergence for Newton's method
    while basin_check > 1:
        fm = f(m)

        if f(m)*f(a) > 0:
            a = m
            fa = fm
        else:
            b = m

        m = 0.5*(a+b)
        fm = f(m)
        basin_check = (ddf(m) * f(m) / df**2) - 1

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

'''Question 6'''

f6 = lambda x: np.exp(x**2 + 7*x - 30)


