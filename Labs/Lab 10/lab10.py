import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import quad

def eval_legendre(n, x):
    """
    Evaluates Legendre polynomials up to order n at a given x using three term recursion.
    Returns an array of length n+1 containing the values of the polynomials.
    """
    p = np.zeros(n+1)
    p[0] = 1
    if n > 0:
        p[1] = x

    for i in range(1, n):
        p[i+1] = ((2*i + 1) * x * p[i] - i * p[i - 1]) / (i + 1)
    
    return p

def coefficient(f, phi_j, phi_j_sq, w, a, b):
    num_integrand = lambda x: phi_j(x) * f(x) * w(x)
    numerator, err = quad(num_integrand, a, b)

    denom_integrand = lambda x: phi_j_sq(x) * w(x)
    denominator, err = quad(denom_integrand, a, b)

    return numerator / denominator

def eval_legendre_expansion(f, a, b, w, n, x):
    """
    Evaluates the Legendre polynomial expansion of f at point x over [a, b] with order n.
    """
    p = eval_legendre(n, x)
    pval = 0.0

    for j in range(n+1):
        phi_j = lambda x: eval_legendre(j, x)[j]
        phi_j_sq = lambda x: phi_j(x) ** 2

        aj = coefficient(f, phi_j, phi_j_sq, w, a, b)
        
        pval += aj*p[j]

    return pval

def driver():
    f = lambda x: math.exp(x)  
    a, b = -1, 1              
    w = lambda x: 1.0          
    n = 2                      
    N = 1000                   

    xeval = np.linspace(a, b, N+1)
    pval = np.array([eval_legendre_expansion(f, a, b, w, n, x) for x in xeval])
    fex = np.array([f(x) for x in xeval])

    plt.figure()
    plt.plot(xeval, fex, label='f(x)')
    plt.plot(xeval, pval, label='Expansion')
    plt.legend()
    plt.show()

    err = np.abs(pval - fex)
    plt.figure()
    plt.semilogy(xeval, err, label='Error Function')
    plt.legend()
    plt.show()

driver()