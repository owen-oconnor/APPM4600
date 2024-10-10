import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

'''Define functions'''

def eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

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

'''Exercises'''

f = lambda x: 1 / (1 + (10*x)**2)

a = -1
b = 1

for N in range(2, 11):
    xint = np.linspace(a, b, N+1)
    yint = f(xint)

    V = Vandermonde(xint, N)
    #print(xint, V, N)
    Vinv = inv(V)
    coeffs = Vinv @ yint # apply inverse of Vandermonde matrix to function values to determine coefficients

    Neval = 1000 
    xeval = np.linspace(a, b, Neval+1)
    yeval = eval_monomial(xeval, coeffs, N, Neval) # evaluate polynomial with calculated coefficients

    yexact = f(xeval)

    #error = np.linalg.norm(yexact - yeval)
    plt.plot(xeval, yexact, label='Exact Function', color='blue')
    plt.plot(xeval, yeval, label='Monomial Approx', linestyle='--', color='green')
    #plt.plot(xeval, error, label='Absolute Error', color='red')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

plt.show()

