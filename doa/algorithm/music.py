import numpy as np

def music(received_data, num_signal):
    # compute corvariance matrix
    matrix_r = 1 / received_data.shape[1] *\
        (received_data @ received_data.transpose().conj())

    _, matrix_e = np.linalg.eigh(matrix_r)  # 特征分解
    # eigenvalues are sorted in ascending order.
    noise_space = matrix_e[:,:-num_signal]

