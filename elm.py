import numpy as np
import math
import scipy
import utils
from custom_score_module import func_standardized_rmse, neg_standardized_rmse
import neural_tangents as nt
from neural_tangents import stax
import time


def ELM(n, activation, seed, X, y):

    n0 = X.shape[1] 
    if seed == -1:
        seed = int(time.time_ns() % 1000000)
    np.random.seed = seed
#    self.n0 = n0
#    self.n = n

    
    W = np.random.normal(0, 1/n0, size=(n0, n))

    Z = X @ W
    if activation =='relu':
        S = np.maximum(Z,0)
    elif activation =='erf':
        S = scipy.special.erf(Z)
    
    coef = np.linalg.inv(S.T @ S) @ S.T @ y
    
    y_hat = S @ coef
    err = func_standardized_rmse(y, y_hat)
    return W, coef, err
    
def score_orthogonal_directions(activation, W, beta, X, y):
    Z = X @ W
    if activation =='relu':
        S = np.maximum(Z,0)
    elif activation =='erf':
        S = scipy.special.erf(Z)

    weighted_S = S * beta.reshape(1,-1)
    y_hats = np.cumsum(weighted_S, axis=1) #compute y_hat for each n without using for loop
    err = np.sqrt(np.sum((y.reshape(-1,1) - y_hats)**2, axis = 0)) / np.sqrt(np.sum((y)**2))
 
    return err

def score(activation, W, beta, X, y):
    Z = X @ W
    if activation =='relu':
        S = np.maximum(Z,0)
    elif activation =='erf':
        S = scipy.special.erf(Z)

    y_hat = S @ beta
    err = func_standardized_rmse(y, y_hat)
    return err

def ENRELM(activation, X, y, subset_by_value, sort_by_correlation, threshold):
        K = gen_K(X.T, X.T, act = activation)

        L,U_unsorted = utils.sortedeigh(K, subset_by_value)

        if sort_by_correlation:
            correlations = np.abs(U_unsorted.T @ y).flatten()
            sorted_indices = np.argsort(correlations)[::-1]
            U = U_unsorted[:, sorted_indices]
        else:
            idx = np.searchsorted(np.cumsum(L) / L.sum(), threshold)  
            U = U_unsorted[:,0:idx]


        W = np.linalg.inv(X.T @ X) @ X.T @ scipy.special.erfinv(U)      # TODO change if erf is not used
    
        beta = U.T @ y
        
        #weighted_U = U * beta.reshape(1,-1)
        #y_hats = np.cumsum(weighted_U, axis=1) #compute y_hat for each n without using for loop


        #err = np.sqrt(np.sum((y.reshape(-1,1) - y_hats)**2, axis = 0)) / np.sqrt(np.sum((y)**2))

        return W, beta#, err

def gen_K(A,B, act = "relu"):
        normA = np.sqrt(np.sum(A**2,axis=0))
        normB = np.sqrt(np.sum(B**2,axis=0))

        AB = A.T @ B
        angle_AB = np.minimum( (1/normA).reshape((len(normA),1)) * AB * (1/normB).reshape( (1,len(normB)) ) ,1.)

        if act=="relu":
            print("Computing relu K...")
            K = 1/(2*math.pi)* normA.reshape((len(normA),1)) * (angle_AB*np.arccos(-angle_AB)+np.sqrt(1-angle_AB**2)) * normB.reshape( (1,len(normB)) )
        elif act =="erf":
          #  print("Computing erf K...")
            K = 2/math.pi*np.arcsin(2*AB/np.sqrt((1+2*(normA**2).reshape((len(normA),1)))*(1+2*(normB**2).reshape((1,len(normB))))))
        return K



