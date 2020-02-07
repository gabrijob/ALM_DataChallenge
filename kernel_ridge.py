import numpy as np
from numpy.linalg import inv

penalty = 0.00001

def gram_matrix(A):
    return A.dot(A.T)

def regression(x,y):
    n = len(x)
    X = np.array(x, np.float)
    Y = np.array(y, np.int)
    K = gram_matrix(X)
    print ("K shape: {0}".format(K.shape))
    beta = np.add(K, penalty * n * np.identity(n))
    print ("Inv shape: {0}".format( beta.shape))
    alpha = inv(beta).dot(Y)
    print("Alpha shape: {0}".format( alpha.shape))
    weights = X.T.dot(alpha)
    print ("Weights shape: {0} ".format( weights.shape))
    return weights

def test(x,weights):
    X = np.array(x, np.float)
    w = np.array(weights, np.float)

    #print(X)
    #print(w)
    predictions = X.dot(weights)

    return predictions

