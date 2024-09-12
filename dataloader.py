import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 

# The following dataset are downloaded from https://archive.ics.uci.edu/datasets


DATA_PATH = DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/datasets'

def load_student():
    data = pd.read_csv(DATA_PATH + '/student/Student/student-mat.csv',delimiter=";", header=None)
    y = data[32].values
    X = data[range(0, 30)]

    # Creating dummies for binary and nominal attributes
    dummies0 = pd.get_dummies(X[0], prefix='school')      # school
    dummies1 = pd.get_dummies(X[1], prefix='sex')         # sex
    dummies3 = pd.get_dummies(X[3], prefix='address')     # address
    dummies4 = pd.get_dummies(X[4], prefix='famsize')     # famsize
    dummies5 = pd.get_dummies(X[5], prefix='Pstatus')     # Pstatus
    dummies8 = pd.get_dummies(X[8], prefix='Mjob')        # Mjob
    dummies9 = pd.get_dummies(X[9], prefix='Fjob')        # Fjob
    dummies10 = pd.get_dummies(X[10], prefix='reason')    # reason
    dummies11 = pd.get_dummies(X[11], prefix='guardian')  # guardian
    dummies15 = pd.get_dummies(X[15], prefix='schoolsup') # schoolsup
    dummies16 = pd.get_dummies(X[16], prefix='famsup')    # famsup
    dummies17 = pd.get_dummies(X[17], prefix='paid')      # paid
    dummies18 = pd.get_dummies(X[18], prefix='activities')# activities
    dummies19 = pd.get_dummies(X[19], prefix='nursery')   # nursery
    dummies20 = pd.get_dummies(X[20], prefix='higher')    # higher
    dummies21 = pd.get_dummies(X[21], prefix='internet')  # internet
    dummies22 = pd.get_dummies(X[22], prefix='romantic')  # romantic

    # Combine all dummies and numerical attributes into a single dataframe
    X = pd.concat([
        dummies0,
        dummies1,
        X[2],  # age
        dummies3,
        dummies4,
        dummies5,
        X[6],  # Medu
        X[7],  # Fedu
        dummies8,
        dummies9,
        dummies10,
        dummies11,
        X[12], # traveltime
        X[13], # studytime
        X[14], # failures
        dummies15,
        dummies16,
        dummies17,
        dummies18,
        dummies19,
        dummies20,
        dummies21,
        dummies22,
        X[24], # famrel
        X[25], # freetime
        X[26], # goout
        X[27], # Dalc
        X[28], # Walc
        X[29]  # health
    ], axis=1)

    return X,y

def load_abalone():
    data = pd.read_csv(DATA_PATH + '/abalone/abalone.data', header=None)
    y = data[8].values.astype(float)
    dummies0 = pd.get_dummies(data[0], prefix='Sex')
    X = pd.concat([
        dummies0,
        data[range(1,8)]
    ], axis=1).values.astype(float)
    return X,y

def load_intrusion_wsns():
    data = pd.read_csv(DATA_PATH + '/intrusion_wsns/Intrusion_Wsns/intrusion_wsns.csv', header=0)
    columns = data.columns
    y = data[columns[4]].values.astype(float)
    dummies0 = pd.get_dummies(data[columns[0]], prefix='Area')
    X = pd.concat([
        dummies0,
        data[columns[range(1,4)]]
    ], axis=1).values.astype(float)
    return X,y

def load_forest_fires():
    data = pd.read_csv(DATA_PATH + '/forest_fires/Forest_Fires/forest_fires.csv', header=0)
    columns = data.columns
    y = data[columns[12]].values.astype(float)
    X = data[columns[range(0, 12)]]

    dummies2 = pd.get_dummies(X[columns[2]], prefix='month')         # month
    dummies3 = pd.get_dummies(X[columns[3]], prefix='day')     # day
    X = pd.concat([
        X[columns[range(0,2)]],  # age
        dummies2,
        dummies3,
        X[columns[range(4,12)]]
    ], axis=1).values.astype(float)

    return X,y

def load_prostate():
    data = pd.read_csv(DATA_PATH + '/prostate/prostate.data.txt', delimiter='\t', header=0)
    data.drop(labels=[data.columns[0], data.columns[10]], inplace=True, axis=1)
    y = data[data.columns[8]].values.astype(float)
    X = data[data.columns[range(0,8)]].values.astype(float)
    return X, y


def load_LAozone():
    data = pd.read_csv(DATA_PATH + '/LAozone/LAozone.data.txt', header=0)
    y = data[data.columns[0]].values.astype(float)
    X = data[data.columns[range(1,10)]].values.astype(float)
    return X, y
    

def load_servo():
    data = pd.read_csv(DATA_PATH + '/servo/Servo/servo.data', header=None)
    y = data[4].values
    X = data[[0, 1, 2, 3]]
    X = pd.get_dummies(X, columns=[0, 1, 2, 3]).values.astype(np.float32)
    return X, y


def load_machine_cpu():
    data = np.genfromtxt(DATA_PATH + '/machine/Machine-Cpu/machine.data',
                         delimiter=',', dtype=str)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)
    return X, y

def load_california_housing():
    data = np.genfromtxt(DATA_PATH + '/cal_housing/CaliforniaHousing/cal_housing.data',
                         delimiter=',', dtype=str)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32) / 100000 #la divisione per 100000 è per avere la stessa y di scikitlearn
    return X, y

def load_delta_ailerons():
    data = np.genfromtxt(DATA_PATH + '/delta_ailerons/Ailerons/delta_ailerons.data',
                         dtype=np.float32)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def load_auto_mpg():
    data = pd.read_csv(DATA_PATH + '/auto/Auto-Mpg/auto.data', header=None)
    data = data.apply(lambda col:
                      col.apply(lambda val:
                                val if val != '?' else None))
    data = data.dropna()
    y = data[7].values.astype(float)
    dummies5 = pd.get_dummies(data[5])
    dummies6 = pd.get_dummies(data[6])
    print(type(data))
    X = pd.concat([data[range(0, 5)], dummies5, dummies6],
                  axis=1).values.astype(float)
    return X, y

def load_bank():
    tr = np.genfromtxt(DATA_PATH + '/bank8FM/Bank/Bank8FM/bank8FM.data',
                          dtype=np.float32)
    te = np.genfromtxt(DATA_PATH + '/bank8FM/Bank/Bank8FM/bank8FM.test',
                          dtype=np.float32)
    data = np.vstack([tr, te])
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH + '/abalone/abalone.data', header=None)
    print(data.head())
    y = data[8].values.astype(float)
    dummies0 = pd.get_dummies(data[0], prefix='Sex')
    X = pd.concat([
        dummies0,
        data[range(1,8)]
    ], axis=1).values.astype(float)
    print(y[0:4])
    print(X[0:4, :])