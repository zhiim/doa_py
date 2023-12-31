import numpy as np

def get_noise_space(received_data, num_signal):
    num_snapshots = received_data.shape[1]

    # compute corvariance matrix
    corvariance_matrix = 1 / num_snapshots *\
        (received_data @ received_data.transpose().conj())

    eigenvalues, eigenvectors = np.linalg.eig(corvariance_matrix)
    sorted_index = np.argsort(np.abs(eigenvalues))  # 由小到大排序的索引
    noise_space = eigenvectors[:, sorted_index[:-num_signal]]

    return noise_space