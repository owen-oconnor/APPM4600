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

def eval_chebychev(n, x):
    """
    Evaluates Chebyshev polynomials up to order n at a given x using three term recursion.
    Returns an array of length n+1 containing the values of the polynomials.
    """   
    T = np.zeros(n+1)
    T[0] = 1

    if n > 0:
        T[1] = x

    for i in range(1, n):
        T[i+1] = 2*x*T[i] - T[i-1]

    return T

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

def eval_chebychev_expansion(f, a, b, w, n, x):
    """
    Evaluates the Legendre polynomial expansion of f at point x over [a, b] with order n.
    """
    p = eval_chebychev(n, x)
    pval = 0.0

    for j in range(n+1):
        phi_j = lambda x: eval_chebychev(j, x)[j]
        phi_j_sq = lambda x: phi_j(x) ** 2

        aj = coefficient(f, phi_j, phi_j_sq, w, a, b)
        
        pval += aj*p[j]

    return pval

def driver():
    f = lambda x: 1 / (1+x**2)  
    a, b = -1, 1              
    w = lambda x: 1.0       
    w2 = lambda x: 1/np.sqrt(1-x**2)   
    n = 5                      
    N = 1000                   

    xeval = np.linspace(a, b, N+1)
    pval_leg = np.array([eval_legendre_expansion(f, a, b, w, n, x) for x in xeval])
    pval_cheb = np.array([eval_chebychev_expansion(f, a, b, w2, n, x) for x in xeval])
    fex = np.array([f(x) for x in xeval])

    plt.figure()
    plt.plot(xeval, fex, label='f(x)')
    plt.plot(xeval, pval_leg, label='Legendre Expansion')
    plt.plot(xeval, pval_cheb, label='Chebychev Expansion')
    plt.title("Exact f(x) vs Legendre vs Chebychev Expansions")
    plt.legend()
    plt.show()

    err_leg = np.abs(pval_leg - fex)
    err_cheb = np.abs(pval_cheb-fex)
    plt.figure()
    plt.semilogy(xeval, err_leg, label='Legendre Error')
    plt.semilogy(xeval, err_cheb, label='Chebycheb Error')
    plt.legend()
    plt.show()

driver()