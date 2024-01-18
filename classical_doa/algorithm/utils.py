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


def get_signal_space(received_data, num_signal):
    num_snapshots = received_data.shape[1]

    # compute corvariance matrix
    corvariance_matrix = 1 / num_snapshots *\
        (received_data @ received_data.transpose().conj())

    eigenvalues, eigenvectors = np.linalg.eig(corvariance_matrix)
    sorted_index = np.argsort(np.abs(eigenvalues))  # 由小到大排序的索引
    noise_space = eigenvectors[:, sorted_index[-num_signal:]]

    return noise_space


def divide_into_fre_bins(received_data, num_groups, fs):
    """Do FFT on array signal of each channel, and divide signal into different
    frequency points.

    Args:
        received_data : array received signal
        num_groups : how many groups divide snapshots into
        fs : sampling frequency

    Returns:
        `signal_fre_bins`: 一个m*n*l维的矩阵, 其中m为阵元数, n为FFT的点数,
            l为组数
        `fre_bins`: 每一个FFT点对应的频点
    """
    num_snapshots = received_data.shape[1]

    n_each_group = num_snapshots // num_groups  # 每一组包含的采样点数
    if n_each_group < 128:
        n_fft = 128  # 如果点数太少, 做FFT时补零
    else:
        n_fft = n_each_group

    signal_fre_bins = np.zeros((received_data.shape[0], n_fft, num_groups),
                               dtype=np.complex_)
    # 每一组独立做FFT
    for group_i in range(num_groups):
        signal_fre_bins[:, :, group_i] = np.fft.fft(
            received_data[:, group_i * n_each_group:(group_i+1) * n_each_group],
            n=n_fft,
            axis=1
            )
    fre_bins = np.fft.fftfreq(n_fft, 1 / fs)

    return signal_fre_bins, fre_bins
