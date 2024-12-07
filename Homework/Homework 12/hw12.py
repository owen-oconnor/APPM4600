import numpy as np

A = np.array([[12, 10, 4],
              [10, 8, -5],
              [4, -5, 3]])
H = np.array([[1, 0, 0],
              [0, 0.9286, 0.3712],
              [0, 0.3712, 0.0714]])

Aprime = H @ A @ H.T
print(Aprime)

'Q3'

def power_method(A, tol=1e-6, max_iter=1000):
    """
    Power method to compute the dominant eigenvalue and eigenvector of a matrix A.
    
    Parameters:
    A (ndarray): Input matrix
    tol (float): tolerance
    max_iter (int): Maximum number of iterations
    
    Returns:
    lambda1 (float): Dominant eigenvalue
    x (ndarray): Corresponding eigenvector
    num_iter (int): Number of iterations until convergence
    """
    n = A.shape[0]
    x = np.random.rand(n) 
    x = x / np.linalg.norm(x) 
    
    lambda_old = 0
    for i in range(max_iter):
        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new)  
        lambda_new = np.dot(x_new.T, A @ x_new) 
        
        if np.abs(lambda_new - lambda_old) < tol:
            return lambda_new, x_new, i + 1  # Converged
        
        x = x_new
        lambda_old = lambda_new
    
    raise ValueError("Power method did not converge within the maximum iterations.")

def hilbert_matrix(n):
    """Generate an n x n Hilbert matrix."""
    return np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)])

# Apply the Power Method for different sizes of Hilbert matrices
results = []
for n in range(4, 21, 4): 
    H = hilbert_matrix(n)
    dominant_eigenvalue, eigenvector, iterations = power_method(H)
    results.append((n, dominant_eigenvalue, iterations))

for result in results:
    n, dominant_eigenvalue, iterations = result
    print(f"Matrix Size: {n}, Dom Eigenvalue: {dominant_eigenvalue}, Iters: {iterations}")


def inverse_power_method(A, tol=1e-6, max_iter=1000):
    n = A.shape[0]
    x = np.random.rand(n)  
    x = x / np.linalg.norm(x)

    lambda_old = 0  # Initialize previous eigenvalue
    for i in range(max_iter):
        # Solve A * y = x for y
        y = np.linalg.solve(A, x)
        # Normalize 
        x_new = y / np.linalg.norm(y)
        # Compute the Rayleigh quotient
        lambda_new = np.dot(x_new.T, A @ x_new)
        
        # Check for convergence
        if np.abs(lambda_new - lambda_old) < tol:
            return lambda_new, x_new, i + 1  # Converged
        
        x = x_new
        lambda_old = lambda_new

    raise ValueError("Inverse power method did not converge within the maximum iterations.")

n = 16 
H = np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)])  # Hilbert matrix
smallest_eigenvalue, eigenvector, iterations = inverse_power_method(H)

print(f"Smallest Eigenvalue: {smallest_eigenvalue}")
print(f"Iterations: {iterations}")
