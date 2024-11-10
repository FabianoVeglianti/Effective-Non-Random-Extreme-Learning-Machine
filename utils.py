import numpy as np
import scipy
from sklearn.model_selection import train_test_split

def sortedeigh(matrix):
  #D,U = scipy.linalg.eigh(matrix, subset_by_value=subset_by_value)
  D, U = np.linalg.eigh(matrix)
  sortedIndices = np.argsort(D)[::-1]
  sortedD = D[sortedIndices]
  sortedU = U[:, sortedIndices]
  return sortedD, sortedU

def standardize(X_train, X_test, y_train, y_test):
  # Apply X_train - mean(X_train) / std(X_train)
  # and   X_test - mean(X_train) / std(X_train)
  m = np.mean(X_train, axis = 0)
  s = np.std(X_train, axis = 0)
  X_train = (X_train - m) / s
  X_test = (X_test - m) / s
  return X_train, X_test, y_train, y_test

def center_y(X_train, X_test, y_train, y_test):
  mean_y_train = np.mean(y_train)
  y_train = y_train - mean_y_train
  y_test = y_test - mean_y_train
  return X_train, X_test, y_train, y_test, mean_y_train

def preprocess(X_train, X_test, y_train, y_test):
  # Standardize X
  # y is centered
  X_train, X_test, y_train, y_test = standardize(X_train, X_test, y_train, y_test)
  X_train, X_test, y_train, y_test, mean_y_train = center_y(X_train, X_test, y_train, y_test)
  return X_train, X_test, y_train, y_test, mean_y_train
  

