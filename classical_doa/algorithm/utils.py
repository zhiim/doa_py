import numpy as np


def get_noise_space(corvariance_matrix, num_signal):
    eigenvalues, eigenvectors = np.linalg.eig(corvariance_matrix)
    sorted_index = np.argsort(np.abs(eigenvalues))  # ascending order
    noise_space = eigenvectors[:, sorted_index[:-num_signal]]

    return noise_space


def get_signal_space(corvariance_matrix, num_signal):
    eigenvalues, eigenvectors = np.linalg.eig(corvariance_matrix)
    sorted_index = np.argsort(np.abs(eigenvalues))  # ascending order
    signal_space = eigenvectors[:, sorted_index[-num_signal:]]

    return signal_space


def divide_into_fre_bins(received_data, num_groups, fs):
    """Do FFT on array signal of each channel, and divide signal into different
    frequency points.

    Args:
        received_data : array received signal
        num_groups : how many groups divide snapshots into
        fs : sampling frequency

    Returns:
        `signal_fre_bins`: a m*n*l tensor, in which m equals to number of
            antennas, n is equals to point of FFT, l is the number of groups
        `fre_bins`: corresponding freqeuncy of each point in FFT output
    """
    num_snapshots = received_data.shape[1]

    # number of sampling points in each group
    n_each_group = num_snapshots // num_groups
    if n_each_group < 128:
        n_fft = 128  # zero padding when sampling points is not enough
    else:
        n_fft = n_each_group

    signal_fre_bins = np.zeros(
        (received_data.shape[0], n_fft, num_groups), dtype=np.complex128
    )
    # do FTT separately in each group
    for group_i in range(num_groups):
        signal_fre_bins[:, :, group_i] = np.fft.fft(
            received_data[
                :, group_i * n_each_group : (group_i + 1) * n_each_group
            ],
            n=n_fft,
            axis=1,
        )
    fre_bins = np.fft.fftfreq(n_fft, 1 / fs)

    return signal_fre_bins, fre_bins
