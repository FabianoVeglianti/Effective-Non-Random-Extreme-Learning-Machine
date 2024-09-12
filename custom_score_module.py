import numpy as np
from sklearn.metrics import make_scorer

def func_standardized_rmse(y_true, y_pred):
    return np.sqrt(np.sum((y_true-y_pred)**2))/np.sqrt(np.sum((y_true)**2))

def func_neg_standardized_rmse(y_true, y_pred):
    return -np.sqrt(np.sum((y_true-y_pred)**2))/np.sqrt(np.sum((y_true)**2))

standardized_rmse = make_scorer(func_standardized_rmse, greater_is_better=False)
neg_standardized_rmse = make_scorer(func_neg_standardized_rmse, greater_is_better=True)