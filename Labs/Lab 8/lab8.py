import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv 


def driver():
    
    f = lambda x: 1 / (1 + (10*x)**2)
    a = -1
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 10
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval[j]) 

    
    err = abs(yeval-fex)
    plt.plot(xeval, err, label='Error Function', color='red')
      
    plt.plot(xeval, yeval, label='Approx Function with Linear Spine', color='blue')
    plt.plot(xeval, fex, label='True Function', color='green')
    plt.legend()
    plt.show()   
       


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

        ind = np.where((xeval > a1) & (xeval <= b1))
        xloc = xeval[ind]
        n = len(xloc)

        yloc = np.zeros(len(xloc))

        for kk in range(n):
            yloc[kk] = line_evaluator(a1, fa1, b1, fb1, xloc[kk])

        yeval[ind] = yloc            

    return yeval
           
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
