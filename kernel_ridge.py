import numpy as np
from numpy.linalg import inv

lambda = 0.00001

def gram_matrix(A):
    return A.dot(A.T)

def regression(x,y):
    n = len(x)
    X = np.array(x, np.float)
    Y = np.array(y, np.float)
    K = gram_matrix(X)
    beta = np.add(K, lambda*n*np.identity(n))
    alpha = inv(beta).dot(Y)

    return alpha
