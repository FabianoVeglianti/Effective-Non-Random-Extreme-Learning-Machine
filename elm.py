import numpy as np
import math
import scipy
import logger
import utils
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from custom_score_module import func_standardized_rmse, neg_standardized_rmse
import neural_tangents as nt
from neural_tangents import stax
import jax.numpy as jnp
import time

class ELM(BaseEstimator):
    def __init__(self, n0, n, activation, W, seed = -1):
        if seed == -1:
            seed = int(time.time_ns() % 1000000)
        np.random.seed = seed
        self.n0 = n0
        self.n = n
        self.activation = activation
        if W is None:
            self.W = np.random.normal(0, 1/n0, size=(n0, n))
        else:
            self.W = W

    def set_parameters(self, params):
        self.n0 = params['n0']
        self.n = params['n']
        self.activation = params['activation']
        self.W = params['W']

    def fit(self, X, y):
        # Learns parameters for output layer
        S = self._hidden_layer(X)
        self.coef_ = np.linalg.inv(S.T @ S) @ S.T @ y
        return self
    
    def predict(self, X):
        # Compute y given X
        S = self._hidden_layer(X)
        return S @ self.coef_

    def score(self, X, y):
        # Compute standardized rmse 
        y_hat = self.predict(X)
        return func_standardized_rmse (y, y_hat)
    
    def _hidden_layer(self, X):
        Z = X @ self.W
        if self.activation =='relu':
            hidden = np.maximum(Z,0)
        elif self.activation =='erf':
            hidden = scipy.special.erf(Z)
        return hidden
    
    def select_W(self, X, y, n_sample):
        # Among n_sample possible choices for W, select the best among them in cross validation
        n0 = self.n0
        n = self.n
        activation = self.activation

        W_list = []
        for i in range(n_sample):
            W_list.append(np.random.normal(0, 1/n0, size=(n0, n)))


        # Define the parameter grid
        param_grid = {
            'n0': [n0],
            'n': [n],
            'activation': [activation],
            'W': W_list
        }

        grid_search = GridSearchCV(estimator=ELM(n0, n, activation, None), param_grid=param_grid, cv=5, scoring = neg_standardized_rmse, n_jobs=-1, refit=False)
        grid_search.fit(X, y)
        

        #self.set_parameters(grid_search.cv_results_['params'][grid_search.best_index_])
        return grid_search.cv_results_

class FastELM(ELM):
    def __init__(self, n0, n, activation, W, Cb, Cw):
        super().__init__(n0, n, activation, W)
        self.Cb = Cb
        self.Cw = Cw

    
    #Override
    def fit(self, U, y):
        # Learns parameters for output layer
        self.coef_ = U.T @ y
        return self
    

    def _hidden_layer_n(self, X, n):
        # Same as _hidden_layer, but it use only n columns of self.W
        Z = X @ self.W[:, 0:n].reshape(-1, n)
        if self.activation =='relu':
            hidden = np.maximum(Z,0)
        elif self.activation =='erf':
            hidden = scipy.special.erf(Z)
        return hidden
    
    def predict_n(self, X, n):
        # Same as predict, but it use only n columns of self.W
        S = self._hidden_layer_n(X,n)
        return S @ (self.coef_[0:n].reshape(n,1))

    def score_n(self, X, y, n):
        # Same as score, but it use only n columns of self.W
        y_hat = self.predict_n(X, n)

        return func_standardized_rmse(y_true=y, y_pred=y_hat)

    def gen_K(self, A,B, act = "relu"):
        normA = np.sqrt(np.sum(A**2,axis=0))
        normB = np.sqrt(np.sum(B**2,axis=0))

        AB = A.T @ B
        angle_AB = np.minimum( (1/normA).reshape((len(normA),1)) * AB * (1/normB).reshape( (1,len(normB)) ) ,1.)

        if act=="relu":
            print("Computing relu K...")
            K = 1/(2*math.pi)* normA.reshape((len(normA),1)) * (angle_AB*np.arccos(-angle_AB)+np.sqrt(1-angle_AB**2)) * normB.reshape( (1,len(normB)) )
        elif act =="erf":
            print("Computing erf K...")
            K = 2/math.pi*np.arcsin(2*AB/np.sqrt((1+2*(normA**2).reshape((len(normA),1)))*(1+2*(normB**2).reshape((1,len(normB))))))
        return K
    
    def get_subspace(self, X, y, subset_by_value=None):
        if subset_by_value is None:
            subset_by_value = [1e-4, np.inf]

    #    if self.activation == "erf":
    #        _,_, kernel_fn = stax.serial(
    #            stax.Dense(self.n, W_std = np.sqrt(self.Cw), b_std = np.sqrt(self.Cb), parameterization = 'standard'),
    #            stax.Erf()
    #        )
    #    else :
    #        logger.error("Function not yet implemented!")

    #    K = np.asarray(kernel_fn(jnp.array(X), None).nngp)
        K = self.gen_K(X.T, X.T, act = self.activation)

        L,U = utils.sortedeigh(K, subset_by_value)


        correlations = np.abs(U.T @ y).flatten()
        sorted_indices = np.argsort(correlations)[::-1]
        U_sorted = U[:, sorted_indices]
    
        return L,U_sorted
    
    def select_W(self, X, U):
        # Select W* using subset of U directions
        self.W = np.linalg.inv(X.T @ X) @ X.T @ scipy.special.erfinv(U)
    
        return self.W

if __name__ == "__main__":
    import neural_tangents
    print(neural_tangents.__version__) 



    