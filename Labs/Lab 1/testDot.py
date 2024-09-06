import numpy as np
import numpy.linalg as la
import math

def driver():
    n = 100
    x = np.linspace(0, np.pi, n)

    f = lambda x: np.sin(x)
    g = lambda x: np.cos(x) # sine and cosine are orthogonal

    y = f(x)
    w = g(x)

    dp = dotProduct(y, w, n)
    print(f'The dot product is {dp}')

    A = np.array([[1,2], [3,4]])
    B = np.array([[5,6], [7,8]])

    matrix_prod = matrixmult(A, B) # test matrix multiplication code with 2x2 matrices
    print(f'The matrix product is {matrix_prod}')
    print(f'The matrix product with built in numpy function is {np.matmul(A,B)}') # evaluate same matrix product using built in numpy function

    C = np.array([[1,2,3], [4,5,6], [7,8,9]])
    D = np.array([[9,8,7], [6,5,4], [3,2,1]]) # use same test for 3x3 matrices
    matrix_prod_3x3 = matrixmult(C,D)
    print(f'The matrix product of C and D is {matrix_prod_3x3}')
    print(f'The matrix product of C/D with built in func is {np.matmul(C,D)}')

    return

def dotProduct(x, y, n):
    dp = 0

    for i in range(n):
        dp += x[i] * y[i]

    return dp

def matrixmult(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    is_rectangular = True

    for r, s in zip(A, B): # check to ensure matrix multiplication is applicable for given matrices
        if len(r) != cols_A or len(s) != cols_B:
            is_rectangular = False

    if cols_A != rows_B or not is_rectangular:
        raise ValueError("Cannot multiply the two matrices A and B. Number of columns in A is not equal to the number of rows in B")

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A): # loops through A and B, computing matrix elements and inserting them into the correct index of the preconstructed result
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

driver()