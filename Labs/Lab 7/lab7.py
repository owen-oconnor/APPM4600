import numpy as np
from numpy.linalg import inv

def eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

   
def Vandermonde(xint,N):

    V = np.zeros((N+1,N+1))
    
    ''' fill the first column'''
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    return V


f = lambda x: 1 / 1 + (10*x)**2

a = -1
b = -1
N = 2

xint = np.linspace(a, b, N+1)
yint = f(xint)

V = Vandermonde(xint, N)
Vinv = inv(V)
coeffs = Vinv @ yint # apply inverse of Vandermonde matrix to function values to determine coefficients

Neval = 1000 
xeval = np.linspace(a, b, Neval)
yeval = eval_monomial(xeval, coeffs, N, Neval) # evaluate polynomial with calculated coefficients