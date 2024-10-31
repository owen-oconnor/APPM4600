import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def eval_legendre(n, x):
    '''Evaluates Legendre Polynomial up to order n at value x
    
        Inputs:
        
        n: order of Legendre Poly
        x: value of evaluation

        Outputs:

        p: vector whose entries are the values of legendre poly at x
    
    '''

    