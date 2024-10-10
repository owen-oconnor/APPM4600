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

Neval = 1000 
xeval = np.linspace(a, b, Neval+1)
yexact = f(xeval)

for N in range(2, 21):
    if N <= 11:
        xint = np.linspace(a, b, N+1)
        yint = f(xint)

        '''Monomial method'''
        V = Vandermonde(xint, N)
        Vinv = inv(V)
        coeffs = Vinv @ yint # apply inverse of Vandermonde matrix to function values to determine coefficients
        yeval_mono = eval_monomial(xeval, coeffs, N, Neval) # evaluate polynomial with calculated coefficients

        '''Lagrange and DD methods'''

        yeval_lagrange = np.zeros(Neval+1)
        yeval_dd = np.zeros(Neval+1)
  
        '''Initialize and populate the first columns of the 
        divided difference matrix. We will pass the x vector'''
    
        y = np.zeros( (N+1, N+1) )
        for j in range(N+1):
            y[j][0]  = yint[j]

        y = dividedDiffTable(xint, y, N+1)
        ''' evaluate lagrange poly '''
        for kk in range(Neval+1):
            yeval_lagrange[kk] = eval_lagrange(xeval[kk],xint,yint,N)
            yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)


        '''Plot functions'''
        plt.plot(xeval, yeval_mono, label=f'Monomial Approx for N = {N}', linestyle='--', color=np.random.rand(3,))
        plt.plot(xeval, yeval_lagrange, label=f'Lagrange Approx for N = {N}', linestyle=':', color=np.random.rand(3,))
        plt.plot(xeval, yeval_dd, label=f'DD Approx for N = {N}', linestyle='-.', color=np.random.rand(3,))

        '''Calculate and plot errors'''
        error_mono = yexact - yeval_mono
        error_lagrange = yexact - yeval_lagrange
        error_dd = yexact - yeval_dd

        plt.plot(xeval, error_mono, label=f'Monomial Error for N = {N}', color='green') # plot errors
        plt.plot(xeval, error_lagrange, label=f'Lagrange Error for N = {N}', color='purple')
        plt.plot(xeval, error_dd, label=f'DD Error for N = {N}', color='red')

plt.plot(xeval, yexact, label='Exact Function', color='blue') # Plot exact function

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Interpolation Methods for N = [2, 10]')
plt.show()

'''Extending beyond N = 10'''

for N in range(11, 21):
    xint = np.linspace(a, b, N+1)
    yint = f(xint)

    '''Monomial method'''
    V = Vandermonde(xint, N)
    Vinv = inv(V)
    coeffs = Vinv @ yint # apply inverse of Vandermonde matrix to function values to determine coefficients
    yeval_mono = eval_monomial(xeval, coeffs, N, Neval) # evaluate polynomial with calculated coefficients

    '''Lagrange and DD methods'''

    yeval_lagrange = np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
        divided difference matrix. We will pass the x vector'''
    
    y = np.zeros( (N+1, N+1) )
    for j in range(N+1):
        y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_lagrange[kk] = eval_lagrange(xeval[kk],xint,yint,N)
        yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)


    '''Plot functions'''
    plt.plot(xeval, yeval_mono, label=f'Monomial Approx for N = {N}', linestyle='--', color=np.random.rand(3,))
    plt.plot(xeval, yeval_lagrange, label=f'Lagrange Approx for N = {N}', linestyle=':', color=np.random.rand(3,))
    plt.plot(xeval, yeval_dd, label=f'DD Approx for N = {N}', linestyle='-.', color=np.random.rand(3,))

plt.plot(xeval, yexact, label='Exact Function', color='blue') # Plot exact function
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolation Methods for N = [11, 20]')
plt.show()

