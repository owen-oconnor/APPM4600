import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

'''Define methods'''

def broyden(f, j, x0 , tol, Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''

    '''Sherman-Morrison 
   (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''


    '''In Broyden 
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''
    
    '''initialize with 1 newton step'''
    
    A0 = j(x0)

    v = f(x0)
    A = inv(A0)

    s = -A.dot(v)
    xk = x0+s
    for  its in range(Nmax):
       '''(save v from previous step)'''
       w = v
       ''' create new v'''
       v = f(xk)
       '''y_k = F(xk)-F(xk-1)'''
       y = v-w;                   
       '''-A_{k-1}^{-1}y_k'''
       z = -A.dot(y)
       ''' p = s_k^tA_{k-1}^{-1}y_k'''
       p = -np.dot(s,z)                 
       u = np.dot(s,A) 
       ''' A = A_k^{-1} via Morrison formula'''
       tmp = s+z
       tmp2 = np.outer(tmp,u)
       A = A+1./p*tmp2
       ''' -A_k^{-1}F(x_k)'''
       s = -A.dot(v)
       xk = xk+s
       if (norm(s)<tol):
          alpha = xk
          ier = 0
          return[alpha,ier,its]
    alpha = xk
    ier = 1
    return[alpha,ier,its]

def lazy_newton(f, j, v0, tol, Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = j(v0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = f(v0)
       v1 = v0 - Jinv.dot(F)
       
       if (norm(v1-v0) < tol):
           vstar = v1
           ier =0
           return[vstar, ier,its]
           
       v0 = v1
    
    vstar = v1
    ier = 1
    return[vstar,ier,its]  

def newton(f, j, x0, tol, Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = j(x0)
       Jinv = inv(J)
       F = f(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return [xstar, ier, its]

def steepest_descent(f, j, v0, tol, Nmax, alpha=0.5):
    for its in range(Nmax):
        F = f(v0)
        J = j(v0)
        v1 = v0 - 2*alpha*J.T @ F

        if norm(v1 - v0) < tol:
            vstar = v1
            return [vstar, its]
        
        v0 = v1

    return [v1, its]


'''Question 1'''

def f1(v):
    x = v[0]
    y = v[1]
    return np.array([x**2 + y**2 - 4,
                     np.exp(x) + y - 1])

def J1(v):
    x = v[0]
    y = v[1]

    J = np.zeros((2,2))

    J[0, 0] = 2*x
    J[0, 1] = 2*y

    J[1, 0] = np.exp(x)
    J[1, 1] = 1

    return J

v1 = np.array([1, 
             1])
v2 = np.array([1, 
               -1])
v3 = np.array([0,
               0])

tol = 1e-10

sol_broyden = broyden(f1, J1, v1, tol, Nmax=500)
sol_lazy = lazy_newton(f1, J1, v1, tol, Nmax=500)

print(f'The approximate solution using Broyden method is {sol_broyden[0]} in {sol_broyden[2]} iterations')
print(f'The approximate solution using the Lazy Newton method is {sol_lazy[0]} in {sol_lazy[2]} iterations')


'''Question 2'''

def f2(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return np.array([x + np.cos(x*y*z) - 1,
                     (1-x)**(1/4) + y + 0.05*z**2 - 0.15*z -1,
                     -x**2 - 0.1*y**2 + 0.01*y + z - 1])

def J2(v):
    x = v[0]
    y = v[1]
    z = v[2]

    J = np.zeros((3,3))

    J[0, 0] = 1 - y*z*np.sin(x*y*z)
    J[0, 1] = - x*z*np.sin(x*y*z)
    J[0, 2] = - x*y*np.sin(x*y*z)

    J[1, 0] = -0.25 * (1-x)**(-3/4)
    J[1, 1] = 1
    J[1, 2] = 0.1*z - 0.15

    J[2, 0] = -2*x
    J[2, 1] = -0.2*y + 0.01
    J[2, 2] = 1

    return J

tol = 5e-2
v2 = np.array([0.5,
              0.5,
              0.5])

sol_steep = steepest_descent(f2, J2, v2, tol, Nmax=500)
print(f2(sol_steep[0]))
print(f'The approximate solution using the Steepest Descent method is {sol_steep[0]} in {sol_steep[1] + 1} iterations')

v2_new = sol_steep[0]
sol_newt = newton(f2, J2, v2_new, tol, Nmax=500)
print(f2(sol_newt[0]))
print(f'The approximate solution using Newtons method is {sol_newt[0]} in {sol_newt[2] + 1} iterations')