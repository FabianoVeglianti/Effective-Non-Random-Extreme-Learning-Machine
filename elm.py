import numpy as np
import math
import scipy
import utils
from custom_score_module import func_standardized_rmse, neg_standardized_rmse
import time


def ELM(n, seed, X, y, test = None):

    n0 = X.shape[0] 
    if seed == -1:
        seed = int(time.time_ns() % 1000000)
    np.random.seed = seed
#    self.n0 = n0
#    self.n = n
    timing = time.time_ns()
    
    W = np.random.normal(0, 1/n0, size=(n, n0))
    
    Z = (W @ X).T
    assert(Z.shape==(X.shape[1], n))
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
    
def score_orthogonal_directions(W, beta, X, y, center_factor = None):
    Z = (W @ X).T 

    S = scipy.special.erf(Z)
    if center_factor is not None:
        S = S - center_factor

    weighted_S = S * beta.reshape(1,-1)
    y_hats = np.cumsum(weighted_S, axis=1) #compute y_hat for each n without using for loop
    err = np.sqrt(np.sum((y.reshape(-1,1) - y_hats)**2, axis = 0)) / np.sqrt(np.sum((y)**2))
 
    return err

def score(W, beta, X, y):
    Z = (W @ X).T 

    S = scipy.special.erf(Z)

    y_hat = S @ beta
    err = func_standardized_rmse(y, y_hat)
    return err

def incremental_ENRELM(X, y, epsilon, threshold, test = None):
    hidden_space_dim = int(np.minimum(50 * X.shape[0], round(X.shape[1] * 0.5)))
    training_error = np.zeros(hidden_space_dim+1)


    timing = time.time_ns()
    K = gen_K(X, X)
    _,U = utils.sortedeigh(K)

    W = scipy.special.erfinv(U.T) @ X.T @ np.linalg.inv(X @ X.T)
    S = scipy.special.erf((W @ X).T)

    mean_S = np.mean(S, axis = 0)
    
    S_ = S - mean_S

    y_hat = np.ones((X.shape[1],1)) * np.mean(y)

    r = y - y_hat 
    training_error[0] = func_standardized_rmse(y, y_hat)



    #score_inf[num_eigenvec_used_inf] = train_error_inf[num_eigenvec_used_inf] + (num_eigenvec_used_inf * np.log(Tr)) * (train_error_inf[0] / Tr / np.log(Tr))
    #beta = np.zeros()
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


        if (old_training_error - training_error[num_dim] < threshold):
            break

    timing = (time.time_ns() - timing) / 1e6

    data = None

    if test is not None:
        X_test = test['X']
        y_test = test['y']
        test_error = np.zeros(len(dim_insert)+1)
        

        W = W[dim_insert, :]
        beta_incremental = beta_incremental[dim_insert]
        y_test_hat = np.ones((X_test.shape[1],1)) * np.mean(y)
        test_error[0] = func_standardized_rmse(y_test, y_test_hat)
        test_error[1:] = score_orthogonal_directions(W, beta_incremental, X_test, y_test, center_factor=mean_S[dim_insert])
        training_error = training_error[training_error!= 0]
        training_error[1:] = score_orthogonal_directions(W, beta_incremental, X, y, center_factor=mean_S[dim_insert])

        data = {
            'training_error' : training_error,
            'test_error' : test_error, 
            'timing': timing 
        }

    return W, beta_incremental, mean_S, data



def approximated_ENRELM(X, y, sort_by_correlation, test = None):
        max_n = int(np.minimum(50 * X.shape[0], round(X.shape[1] * 0.5)))
        timing = time.time_ns()
        K = gen_K(X, X)

        L,U_unsorted = utils.sortedeigh(K)

        if sort_by_correlation:
            correlations = np.abs(U_unsorted.T @ y).flatten()
            sorted_indices = np.argsort(correlations)[::-1]
            U = U_unsorted[:, sorted_indices]
        else:  
            U = U_unsorted

        W = scipy.special.erfinv((U[:, 0:max_n]).T) @ X.T @ np.linalg.inv(X @ X.T)
    
        beta = (U[:, 0:max_n]).T @ y
        
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


def gen_K(A,B):
        normA = np.sqrt(np.sum(A**2,axis=0))
        normB = np.sqrt(np.sum(B**2,axis=0))

        AB = A.T @ B
        angle_AB = np.minimum( (1/normA).reshape((len(normA),1)) * AB * (1/normB).reshape( (1,len(normB)) ) ,1.)


        #  print("Computing erf K...")
        K = 2/math.pi*np.arcsin(2*AB/np.sqrt((1+2*(normA**2).reshape((len(normA),1)))*(1+2*(normB**2).reshape((1,len(normB))))))
        return K



