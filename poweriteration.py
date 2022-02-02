### Implementation of the Power Iteration
# for Numerical Methods 2, WS2021/22
# at University of Vienna
# by Clemens Wager

import numpy as np


def power_iteration(A, z):
    """
    > This algorithm computes the eigenvector associated with the largest eigenvalue.
    > A random initial vector is defined to start as candidate eigenvector.
    Input:
        A: input matrix
        z: initial guess vector
    Return:
        z: the computed eigenvector
        n: matrix dimension
    """
    n = len(A)
    print(f"matrix dimension n = {n}")
    print(f"original vector z =\n{z}")
    print(f"original matrix A =\n{A}")

    for i in range(5): # loop 5 times 
        print(f"\nRUN {i+1}")
        # calculate the matrix-by-vector product Ab
        Az = np.dot(A, z)
        print(f"Az = {Az}")

        # calculate the norm
        Az_norm = np.linalg.norm(Az)
        print(f"Az_norm = {Az_norm}")

        # re normalize the vector
        z = Az / Az_norm
        print(f"Az normalized = {z}")

    return z, n


# example from internet
A2 = np.array(
     [[0.5, 0.5],
     [0.2, 0.8]])

# task 2 example
A3 = np.array(
     [[1., 0., 0.],
     [0., 0., 0.],
     [0., 0., -1.]])

# define given vector
v = np.array([1 / np.sqrt(3)] * 3)

# call power iteration function
v_tilde, n = power_iteration(A3, v)

# output
print(f"\n-------- RESULT ---------")
v_tilde = np.reshape(v_tilde, (n,1))
print(f"associated eigenvector =\n{v_tilde}")


"""
Observation:
The z-coordinate of the resulting eigenvector alternates.
Odd iteration number: negative z-coordinate
Even iteration number: positive z-coordinate
"""
print("\n#####################")
print(np.sqrt((2/9)))
a = np.linalg.norm(A3 @ v)
print(a)
print((np.sqrt( (1 / np.sqrt(3)) **2
                +
                (-1 / np.sqrt(3)) **2 ) ))
