import numpy as np
import math
import scipy
import utils
from custom_score_module import func_standardized_rmse, neg_standardized_rmse
import neural_tangents as nt
from neural_tangents import stax
import time


def ELM(n, seed, X, y, test = None):

    n0 = X.shape[1] 
    if seed == -1:
        seed = int(time.time_ns() % 1000000)
    np.random.seed = seed
#    self.n0 = n0
#    self.n = n
    timing = time.time_ns()
    
    W = np.random.normal(0, 1/n0, size=(n0, n))

    Z = X @ W

    S = scipy.special.erf(Z)
    
    beta = np.linalg.inv(S.T @ S) @ S.T @ y
    
    if test is not None:
        X_test = test['X']
        y_test = test['y']
        training_error = score(W, beta, X, y)
        test_error = score(W, beta, X_test, y_test)


    timing = (time.time_ns() - timing)/1e6
    data = None
    if test is not None:
        data = {
            'training_error' : training_error,
            'test_error' : test_error, 
            'timing': timing  
        } 
    return W, beta, data
    
def score_orthogonal_directions(W, beta, X, y):
    Z = X @ W

    S = scipy.special.erf(Z)

    weighted_S = S * beta.reshape(1,-1)
    y_hats = np.cumsum(weighted_S, axis=1) #compute y_hat for each n without using for loop
    err = np.sqrt(np.sum((y.reshape(-1,1) - y_hats)**2, axis = 0)) / np.sqrt(np.sum((y)**2))
 
    return err

def score(W, beta, X, y):
    Z = X @ W

    S = scipy.special.erf(Z)

    y_hat = S @ beta
    err = func_standardized_rmse(y, y_hat)
    return err

def incremental_ENRELM(X, y, threshold, test = None):
    hidden_space_dim = int(np.minimum(50 * X.shape[1], round(X.shape[0] * 0.5)))
    training_error = np.zeros(hidden_space_dim+1)

    test_is_not_None = test is not None
    if test_is_not_None:
        X_test = test['X']
        y_test = test['y']
        test_error = np.zeros(hidden_space_dim+1)


    timing = time.time_ns()
    K = gen_K(X.T, X.T)
    L,U = utils.sortedeigh(K, None)

    W = np.linalg.inv(X.T @ X) @ X.T @ scipy.special.erfinv(U)    
    S = scipy.special.erf(X @ W)

    mean_S = np.mean(S, axis = 0)
    
    S_ = S - mean_S

    y_hat = np.ones((X.shape[0],1)) * np.mean(y)

    r = y - y_hat 
    training_error[0] = func_standardized_rmse(y, y_hat)

    if test_is_not_None:
        y_test_hat = np.ones((X_test.shape[0],1)) * np.mean(y)
        test_error[0] = func_standardized_rmse(y_test, y_test_hat)
        S_test = scipy.special.erf(X_test @ W)
        S_test_ = S_test - mean_S


    #score_inf[num_eigenvec_used_inf] = train_error_inf[num_eigenvec_used_inf] + (num_eigenvec_used_inf * np.log(Tr)) * (train_error_inf[0] / Tr / np.log(Tr))
    #beta = np.zeros()
    epsilon = 1 / np.sqrt(y.shape[0])
    dim_insert = []
    num_dim = 0
    iteration = 0
    beta_incremental = np.zeros(S_.shape[1])
    while(num_dim <= hidden_space_dim):
        dotprod = np.dot(S_.T, r)
        absdotprod = np.abs(dotprod)
        l = np.argmax(absdotprod)

        old_training_error = training_error[num_dim]
        if not l in dim_insert:
            num_dim += 1
            dim_insert.append(l)
        
        
        
        beta_incremental[l] = beta_incremental[l] + epsilon * dotprod[l] / (np.linalg.norm(S_[:, l:l+1]))**2
        y_hat = y_hat + epsilon * dotprod[l] / (np.linalg.norm(S_[:, l:l+1]))**2 * S_[:, l:l+1]

        r = r - epsilon * dotprod[l] / (np.linalg.norm(S_[:, l:l+1]))**2 * S_[:, l:l+1]
        training_error[num_dim] = func_standardized_rmse(y, y_hat)

        if test_is_not_None:
            y_test_hat = y_test_hat + epsilon * dotprod[l] / (np.linalg.norm(S_[:, l:l+1]))**2 * S_test_[:, l:l+1] 
            test_error[num_dim] = func_standardized_rmse(y_test, y_test_hat)

        if (old_training_error - training_error[num_dim] < threshold):
            break

    timing = (time.time_ns() - timing) / 1e6

    data = None
    if test_is_not_None:
        data = {
            'training_error' : training_error,
            'test_error' : test_error, 
            'timing': timing 
        }

    return W, beta_incremental, mean_S, data



def approximated_ENRELM(X, y, sort_by_correlation, threshold, test = None):
        timing = time.time_ns()
        K = gen_K(X.T, X.T)

        L,U_unsorted = utils.sortedeigh(K)

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

        if test is not None:
            X_test = test['X']
            y_test = test['y']
            training_error = score_orthogonal_directions(W, beta, X, y)
            test_error = score_orthogonal_directions(W, beta, X_test, y_test)
 
  
        timing = (time.time_ns() - timing)/1e6
        data = None
        if test is not None:
            data = {
                'training_error' : training_error,
                'test_error' : test_error, 
                'timing': timing  
            }   
        return W, beta, data


def ENRELM(X, y, sort_by_correlation, threshold, max_dim):
    K = gen_K(X.T, X.T)

    L,U_unsorted = utils.sortedeigh(K)

    if sort_by_correlation:
        correlations = np.abs(U_unsorted.T @ y).flatten()
        sorted_indices = np.argsort(correlations)[::-1]
        U = U_unsorted[:, sorted_indices]
    else:
        idx = np.searchsorted(np.cumsum(L) / L.sum(), threshold)  
        U = U_unsorted[:,0:idx]


    W = np.linalg.inv(X.T @ X) @ X.T @ scipy.special.erfinv(U)      # TODO change if erf is not used

    beta = np.zeros((max_dim, max_dim))
    for j in range(max_dim):
        S = scipy.special.erf(X @ W[:,0:j+1])
        beta[0:j+1,j:j+1] = np.linalg.inv(S.T @ S) @ S.T @ y
    
    #weighted_U = U * beta.reshape(1,-1)
    #y_hats = np.cumsum(weighted_U, axis=1) #compute y_hat for each n without using for loop


    #err = np.sqrt(np.sum((y.reshape(-1,1) - y_hats)**2, axis = 0)) / np.sqrt(np.sum((y)**2))

    return W, beta#, err

def gen_K(A,B):
        normA = np.sqrt(np.sum(A**2,axis=0))
        normB = np.sqrt(np.sum(B**2,axis=0))

        AB = A.T @ B
        angle_AB = np.minimum( (1/normA).reshape((len(normA),1)) * AB * (1/normB).reshape( (1,len(normB)) ) ,1.)


        #  print("Computing erf K...")
        K = 2/math.pi*np.arcsin(2*AB/np.sqrt((1+2*(normA**2).reshape((len(normA),1)))*(1+2*(normB**2).reshape((1,len(normB))))))
        return K



