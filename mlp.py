import tensorflow as tf
import math
import time
from sklearn.metrics import mean_squared_error

# Define the erf activation function
def erf_activation(x):
    return tf.math.erf(x)


def create_mlp(n_list, activation, Cb, Cw, seeds, use_bias = True):
    """
    Args:
        use_bias (bool): attiva/disattiva l'uso dei bias nei layers
        use_generator_init (bool):
        se False usa l'inizializzazione di Gianluca
        se True usa l'inizializzazione descritta in Delving Deep into Rectifiers
    """
    model = tf.keras.Sequential()
    
    L = 1
    #Creo gli inizializzatori -> oggetti necessari a inizializzare i pesi W (chiamati kernel) e i bias
    kernel_initializers_list = []
    bias_initializers_list = []

    kernel_initializers_list.append(tf.keras.initializers.RandomNormal(0, math.sqrt(Cw/n_list[0]), seed = seeds[0]))

    bias_initializers_list.append(tf.keras.initializers.RandomNormal(0,stddev=math.sqrt(Cb), seed = seeds[1]))



    kernel_initializers_list.append(tf.keras.initializers.RandomNormal(0, math.sqrt(Cw/n_list[L]), seed = seeds[2]))

    bias_initializers_list.append(tf.keras.initializers.RandomNormal(0,stddev=math.sqrt(Cb), seed = seeds[3]))


    if activation == "erf":
        activation = erf_activation


    #Aggiungo gli strati al modello: Dense e' il classico strato pieno di neuroni
    model.add(tf.keras.layers.Input(shape=(n_list[0],)))
    model.add(tf.keras.layers.Dense(n_list[1], activation = activation, use_bias = use_bias, kernel_initializer = kernel_initializers_list[0],
                bias_initializer = bias_initializers_list[0]))
    model.add(tf.keras.layers.Dense(n_list[L+1], use_bias = use_bias, kernel_initializer = kernel_initializers_list[L],
                    bias_initializer = bias_initializers_list[L]))
    #print("Created MLP")
    #print(self.summary())
    return model

def train_mlp(model, X_train, y_train, learning_rate, epochs, batch_size,save_gradients=False):
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test, verbose = 0)
    mse = mean_squared_error(y_test, y_pred)

    return y_pred, mse

def evaluate(model, X, y):
    y_pred = model.predict(X, verbose = 0)
    mse = mean_squared_error(y, y_pred)
    return mse