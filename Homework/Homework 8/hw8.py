import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():
    
    f = lambda x: 1./(1. + (10*x)**2)
    fd = lambda x: -20*x/ (1. + (10*x)**2)**2
    a, b = -5, 5

    '''LAGRANGE'''
    
    for Nint in range(5, 21, 5):
        xint = np.linspace(a, b, Nint+1)
        yint = f(xint)
        ydint = fd(xint)

        xint_cheb = (a + b) / 2 + (b - a) / 2 * np.array([np.cos(np.pi*(2*j + 1) / (2*(Nint+1))) for j in range(Nint+1)])
        yint_cheb = f(xint_cheb)
        ydint_cheb = fd(xint_cheb)

        Neval = 100
        xeval = np.linspace(a, b, Neval+1)
        yevalL = np.zeros(Neval+1)
        yevalH = np.zeros(Neval+1)
        yevalL_cheb = np.zeros(Neval+1)
        yevalH_cheb = np.zeros(Neval+1)
        for kk in range(Neval+1):
            yevalL[kk] = eval_lagrange(xeval[kk],xint,yint,Nint)
            yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ydint,Nint)
            yevalL_cheb[kk] = eval_lagrange(xeval[kk],xint_cheb,yint_cheb,Nint)
            yevalH_cheb[kk] = eval_hermite(xeval[kk],xint_cheb,yint_cheb,ydint_cheb,Nint)


        '''CUBIC SPLINE???'''

        '''(M,C,D) = create_natural_spline(yint,xint,Nint)
        #  evaluate the cubic spline     
        yeval_cub = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)'''

        ''' create vector with exact values'''
        fex = f(xeval)
    
    
        plt.plot(xeval,fex,color='red', label='True Function')
        plt.plot(xeval,yevalL,color='blue',label=f'Lagrange Int') 
        plt.plot(xeval,yevalH,color='black',label=f'Hermite Int')

        plt.plot(xeval, yevalL_cheb, color='purple', label=f'Lagrange Int w/ Chebychev')
        plt.plot(xeval, yevalH_cheb, color='green', label='Hermite Int w/ Chebychev')

        #plt.plot(xeval,yeval_cub,'bs--',label='natural cubic spline') 

        plt.title(f'Lagrange and Hermite Int for n={Nint}')
        plt.semilogy()
        plt.legend()
        plt.show()
         
        errL = abs(yevalL-fex)
        errH = abs(yevalH-fex)
        errL_cheb = abs(yevalL_cheb - fex)
        errH_cheb = abs(yevalH_cheb - fex)

        plt.semilogy(xeval,errL,color='blue',label=f'Lagrange Error') 
        plt.semilogy(xeval,errH,color='black',label=f'Hermite Error')
        plt.semilogy(xeval, errL_cheb, color='purple', label=f'Lagrange w/ Cheb Error')
        plt.semilogy(xeval, errH_cheb, color='green', label='Hermite w/ Cheb Error')

        plt.title(f'Errors of Lagrange and Hermite for n={Nint}') 
        plt.legend()
        plt.show()   



def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
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
  
    return yeval

def line_evaluator(x0, f0, x1, f1, alpha):
    return f0 + (f1 - f0) * (alpha - x0) / (x1 - x0)
  
    
def eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 

    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)

        ind = np.where((xeval > a1) & (xeval <= b1)) # find indicies in interval [a1, b1]
        xloc = xeval[ind] # store subset of points in interval [a1, b1]
        n = len(xloc)

        yloc = np.zeros(len(xloc))

        for kk in range(n): # loop over all xeval points in interval [a1, b1] and plot y for these points
            yloc[kk] = line_evaluator(a1, fa1, b1, fb1, xloc[kk])

        yeval[ind] = yloc # add spline to overall y plot for this nodes         

    return yeval
    
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1) 
    for i in range(1,N):
       h[i-1] = xint[i] - xint[i-1]
       b[i] = (yint[i+1]-yint[i])/h[i] - (yint[i]-yint[i-1])/h[i-1]

#  create the matrix A so you can solve for the M values
    A = np.zeros((N+1,N+1))

#  Invert A    
    Ainv = inv(A)

# solver for M    
    M  = 0
    
#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = 0 # find the C coefficients
       D[j] = 0 # find the D coefficients
    return(M,C,D)
       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
   
    yeval = 0
    return yeval 
    
    
def eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#   copy into yeval
        yeval[ind] = yloc

    return(yeval) 
           
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
