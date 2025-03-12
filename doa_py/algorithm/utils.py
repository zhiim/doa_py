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


def divide_into_fre_bins(
    received_data, num_groups, fs, f_min=None, f_max=None, n_fft_min=128
):
    """Do FFT on array signal of each channel, and divide signal into different
    frequency points.

    Args:
        received_data : array received signal
        num_groups : how many groups divide snapshots into
        fs : sampling frequency
        f_min: minimum frequency of interest
        f_max: maximum frequency of interest
        min_n_fft: minimum number of FFT points

    Returns:
        `signal_fre_bins`: a (m, n, l) tensor, in which m equals to number of
            antennas, n is equals to point of FFT, l is the number of groups
        `fre_bins`: corresponding freqeuncy of each point in FFT output
    """
    num_snapshots = received_data.shape[1]

    # number of sampling points in each group
    n_each_group = num_snapshots // num_groups
    if n_each_group < n_fft_min:
        n_fft = n_fft_min  # zero padding when sampling points is not enough
    else:
        n_fft = n_each_group

    delta_f = fs / n_fft
    # there is a little trick to use as wider frequency range as possible
    idx_f_min = max(int(f_min / delta_f) - 1, 0) if f_min is not None else 0
    idx_f_max = (
        min(int(f_max / delta_f) + 1, n_fft // 2)
        if f_max is not None
        else n_fft // 2
    )
    idx_range = idx_f_max - idx_f_min + 1

    signal_fre_bins = np.zeros(
        (received_data.shape[0], idx_range, num_groups),
        dtype=np.complex128,
    )
    # do FTT separately in each group
    for group_i in range(num_groups):
        signal_fre_bins[:, :, group_i] = np.fft.fft(
            received_data[
                :, group_i * n_each_group : (group_i + 1) * n_each_group
            ],
            n=n_fft,
            axis=1,
        )[:, idx_f_min : idx_f_max + 1]
    fre_bins = np.fft.fftfreq(n_fft, 1 / fs)[idx_f_min : idx_f_max + 1]

    return signal_fre_bins, fre_bins


def forward_backward_smoothing(received_data, subarray_size):
    num_elements = received_data.shape[0]
    num_subarrays = num_elements - subarray_size + 1
    smoothed_data = np.zeros(
        (subarray_size, subarray_size), dtype=np.complex128
    )

    # forward smoothing
    for i in range(num_subarrays):
        subarray = received_data[i : i + subarray_size, :]
        smoothed_data += np.cov(subarray)

    # backward smoothing
    matrix_j = np.fliplr(np.eye(subarray_size))
    smoothed_data += matrix_j @ smoothed_data.conj() @ matrix_j

    smoothed_data /= 2 * num_subarrays

    return smoothed_data
