import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
from time import time


def driver():
    ''' Test solving square systems with different solvers and measure performance '''
    
    sizes = [100, 500, 1000, 2000, 4000, 5000]
    num_rhs = 10 

    results = [] 
    
    for N in sizes:
        # Generate random matrix and right-hand sides
        A = np.random.rand(N, N)
        B = np.random.rand(N, num_rhs)
        
        start_normal = time()
        X = scila.solve(A, B)
        time_normal = time() - start_normal
        
        # Solve with LU
        start_lu_fact = time()
        lu, piv = scila.lu_factor(A)
        time_lu_fact = time() - start_lu_fact
        
        start_lu_solve = time()
        X_lu = scila.lu_solve((lu, piv), B)
        time_lu_solve = time() - start_lu_solve
        
        results.append((N, time_normal, time_lu_fact, time_lu_solve))
    
    plot_results(results)


def create_rect(N, M):
    ''' Create an ill-conditioned rectangular matrix '''
    a = np.linspace(1, 10, M)
    d = 10**(-a)
    
    D2 = np.zeros((N, M))
    for j in range(0, M):
        D2[j, j] = d[j]
    
    # Create matrices needed to manufacture the low rank matrix
    A = np.random.rand(N, N)
    Q1, _ = la.qr(A)
    
    A = np.random.rand(M, M)
    Q2, _ = la.qr(A)
    
    B = np.matmul(Q1, D2)
    B = np.matmul(B, Q2)
    return B


def plot_results(results):
    ''' Plot timing results for different solvers '''
    sizes, t_normal, t_lu_fact, t_lu_solve = zip(*results)
    
    plt.plot(sizes, t_normal, label="Normal Solve")
    plt.plot(sizes, t_lu_fact, label="LU Factorization")
    plt.plot(sizes, t_lu_solve, label="LU Solve")
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Time (seconds)")
    plt.title("Performance")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    driver()
