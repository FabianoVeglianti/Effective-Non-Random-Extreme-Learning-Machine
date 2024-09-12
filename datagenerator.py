import numpy as np
import pandas as pd
import os



class DataGenerator:

    def generate(self):
        raise NotImplementedError

    def dataset_name(self):
        raise NotImplementedError
    
    def generate_Toeplix(self, n0, rho):
        matrix = np.zeros((n0, n0))
        for i in range(n0):
            for j in range(n0):
                matrix[i, j] = rho ** abs(i - j)
        return matrix

    def generate_full(self, n0):
        matrix = np.uniform(-2 * np.pi, 2 * np.pi, size = (n0, n0))
        matrix = matrix + matrix.T
        D, U = np.linalg.eigh(matrix)
        cov = U * (np.abs(D)+1e-2) * U.T
        return cov


    def generate_X(self, method, rho):
        if method =="Toeplix":
            cov = self.generate_Toeplix(self.n0, rho)
        if method =="full":
            cov = self.generate_full(self.n0)
        if method =="iid":
            cov = np.eye(self.n0)
        X = np.random.multivariate_normal(0, cov, size = self.T)
        return X



class LoaderDataGenerator(DataGenerator):

    def __init__(self, name, loader_fn):
        self.name = name
        self.loader_fn = loader_fn

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = self.loader_fn()
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class SyntheticData(DataGenerator):
    idx = 0

    def __init__(self, T, n0, X_distribution, X_range, X_cov, X_rho, y_function, y_neurons, y_SNR):
        self.T = T
        self.n0 = n0
        self.X_distribution = X_distribution
        self.X_range = X_range
        self.X_cov = X_cov
        self.X_rho = X_rho
        self.y_function = y_function
        self.y_neurons = y_neurons
        self.y_SNR = y_SNR
        SyntheticData.idx = SyntheticData.idx + 1
        self.name = str(SyntheticData.idx)

    def dataset_name(self):
        return self.name
    
    def generate(self):
        if X_distribution == "uniform":
            X = np.random.uniform(X_range[0], X_range[1], (T, n0))
        elif X_distribution == "gaussian":
            if X_cov == "toeplix":
                cov_matrix = np.fromfunction(
                        lambda i, j: X_rho ** np.abs(i - j), 
                        (n0, n0), 
                        dtype=int
                    )
            elif X_cov == "iid":
                cov_matrix = np.eye(n0)
            X = np.random.multivariate_normal(np.zeros(n0), cov_matrix, T)
        
        # Generate y
        if y_function == "linear":
            weights = np.random.uniform(-2, 2, n0)
            bias = np.random.uniform(-2, 2)
            y = X @ weights + bias
        
        elif y_function == "shallow":
            W_1 = np.random.normal(0, 1/np.sqrt(n0), (n0, self.y_neurons))
            W_2 = np.random.normal(0, 1/np.sqrt(self.y_neurons), (self.y_neurons, 1))
            y = 1/(1 + np.exp(-(X @ W_1))) @ W_2 #shallow NN with sigmoid activation function
        
        
        # Add noise
        var_y = np.var(y)
        var_noise = var_y / y_SNR
        noise = np.random.normal(0, np.sqrt(var_noise), y.shape)
        y += noise

        return X, y

if __name__ == "__main__":

    T = 300
    n0_list = [30,100]
    X_distribution_list = ["uniform", "gaussian"]
    X_range_list = [[-2 * np.pi, 2 * np.pi]]
    X_cov_list = ["toeplix", "iid"]
    X_rho_list = [0.8]
    y_function_list = ["linear", "shallow"]
    #y_neurons_list = [1, 10, 100]
    y_neurons_list = [100]
    y_SNR_list = [2, 10]

    datasets = {}
    index = 1
    for n0 in n0_list:
        for X_distribution in X_distribution_list:
            if X_distribution == "uniform":
                for X_range in X_range_list:
                    for y_function in y_function_list:
                        if y_function == "linear":
                            for y_SNR in y_SNR_list:
                                syntheticDataGenerator = SyntheticData(T, n0, X_distribution, X_range, None, None, y_function, None, y_SNR)
                                X, y = syntheticDataGenerator.generate()
                                datasets[index] = {'X': X,
                                                    'y':y,
                                                    'n0': n0,
                                                    'X_distribution': X_distribution,
                                                    'X_range': X_range,
                                                    'X_cov': "//",
                                                    'X_rho': "//",
                                                    'y_function': y_function,
                                                    'y_neurons': "//",
                                                    'y_SNR': y_SNR}
                                index = index + 1
                        elif y_function == "shallow":
                            for y_neurons in y_neurons_list:
                                for y_SNR in y_SNR_list:
                                    syntheticDataGenerator = SyntheticData(T, n0, X_distribution, X_range, None, None, y_function, y_neurons, y_SNR)
                                    X, y = syntheticDataGenerator.generate()
                                    datasets[index] = {'X': X,
                                            'y':y,
                                            'n0': n0,
                                            'X_distribution': X_distribution,
                                            'X_range': X_range,
                                            'X_cov': "//",
                                            'X_rho': "//",
                                            'y_function': y_function,
                                            'y_neurons': y_neurons,
                                            'y_SNR': y_SNR}
                                    index = index + 1
            elif X_distribution == "gaussian":
                for X_cov in X_cov_list:
                    if X_cov == "iid":
                        for X_range in X_range_list:
                            for y_function in y_function_list:
                                if y_function == "linear":
                                    for y_SNR in y_SNR_list:
                                        syntheticDataGenerator = SyntheticData(T, n0, X_distribution, X_range, None, None, y_function, None, y_SNR)
                                        X, y = syntheticDataGenerator.generate()
                                        datasets[index] = {'X': X,
                                            'y':y,
                                            'n0': n0,
                                            'X_distribution': X_distribution,
                                            'X_range': "//",
                                            'X_cov': X_cov,
                                            'X_rho': "//",
                                            'y_function': y_function,
                                            'y_neurons': "//",
                                            'y_SNR': y_SNR}
                                        index = index + 1
                                elif y_function == "shallow":
                                    for y_neurons in y_neurons_list:
                                        for y_SNR in y_SNR_list:
                                            syntheticDataGenerator = SyntheticData(T, n0, X_distribution, X_range, None, None, y_function, y_neurons, y_SNR)
                                            X, y = syntheticDataGenerator.generate()
                                            datasets[index] = {'X': X,
                                                'y':y,
                                                'n0': n0,
                                                'X_distribution': X_distribution,
                                                'X_range': "//",
                                                'X_cov': X_cov,
                                                'X_rho': "//",
                                                'y_function': y_function,
                                                'y_neurons': y_neurons,
                                                'y_SNR': y_SNR}
                                            index = index + 1
                    elif X_cov == "toeplix":
                        for X_rho in X_rho_list:
                            for X_range in X_range_list:
                                for y_function in y_function_list:
                                    if y_function == "linear" or y_function == "quadratic":
                                        for y_SNR in y_SNR_list:
                                            syntheticDataGenerator = SyntheticData(T, n0, X_distribution, X_range, None, None, y_function, None, y_SNR)
                                            X, y = syntheticDataGenerator.generate()
                                            datasets[index] = {'X': X,
                                                'y':y,
                                                'n0': n0,
                                                'X_distribution': X_distribution,
                                                'X_range': "//",
                                                'X_cov': X_cov,
                                                'X_rho': X_rho,
                                                'y_function': y_function,
                                                'y_neurons': "//",
                                                'y_SNR': y_SNR}
                                            index = index + 1
                                    elif y_function == "shallow":
                                        for y_neurons in y_neurons_list:
                                            for y_SNR in y_SNR_list:
                                                syntheticDataGenerator = SyntheticData(T, n0, X_distribution, X_range, None, None, y_function, y_neurons, y_SNR)
                                                X, y = syntheticDataGenerator.generate()
                                                datasets[index] = {'X': X,
                                                    'y':y,
                                                    'n0': n0,
                                                    'X_distribution': X_distribution,
                                                    'X_range': "//",
                                                    'X_cov': X_cov,
                                                    'X_rho': X_rho,
                                                    'y_function': y_function,
                                                    'y_neurons': y_neurons,
                                                    'y_SNR': y_SNR}
                                                index = index + 1
    # Create a directory to save the dataset files
    os.makedirs('datasets/synthetic', exist_ok=True)

    # Initialize an empty list to hold the summary table data
    synthetic_datasets_description_data = []

    for idx, data in datasets.items():
        # Extract X and y from the dataset
        X = data['X']
        y = data['y']
        
        # Concatenate X and y to form the dataset to be saved
        dataset = np.column_stack((X, y))
        
        # Generate the file name
        file_name = f'dataset_{idx}.csv'
        
        # Save the dataset to a .csv file
        np.savetxt(os.path.join('datasets/synthetic', file_name), dataset, delimiter=',', fmt='%.6f')
        
        # Append the summary information to the summary table data
        synthetic_datasets_description_data.append([
            idx,
            data['n0'],
            data['X_distribution'],
            data['X_range'],
            data['X_cov'],
            data['X_rho'],
            data['y_function'],
            data['y_neurons'],
            data['y_SNR']
        ])

    # Create a DataFrame for the summary table
    summary_df = pd.DataFrame(synthetic_datasets_description_data, columns=[
        'index', 'n0', 'X_distribution', 'X_range', 'X_cov', 'X_rho', 'y_function', 'y_neurons', 'y_SNR'
    ])

    # Save the summary table to a .csv file
    summary_df.to_csv('synthetic_datasets_description.csv', index=False, sep=";")
