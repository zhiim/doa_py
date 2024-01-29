import numpy as np

from classical_doa.algorithm.utils import get_signal_space

C = 3e8


def esprit(received_data, num_signal, array, signal_fre, unit="deg"):
    """Total least-squares ESPRIT. Most names of matrix are taken directly from
    the reference paper.

    Args:
        received_data : Array received signals
        num_signal : Number of signals
        array : Instance of array class
        signal_fre: Signal frequency
        unit : Unit of angle, 'rad' for radians, 'deg' for degrees. Defaults to
            'deg'.

    Reference:
        Roy, R., and T. Kailath. “ESPRIT-Estimation of Signal Parameters via
        Rotational Invariance Techniques.” IEEE Transactions on Acoustics,
        Speech, and Signal Processing 37, no. 7 (July 1989): 984-95.
        https://doi.org/10.1109/29.32276.
    """
    signal_space = get_signal_space(received_data, num_signal)

    # get signal space of two sub array. Each sub array consists of M-1 antennas
    matrix_e_x = signal_space[:-1, :]
    matrix_e_y = signal_space[1:, :]
    # the fixed distance of corresponding elements in two sub-array ensures
    # the rotational invariance
    sub_array_spacing = array.array_position[1] - array.array_position[0]

    matrix_c = np.hstack((matrix_e_x, matrix_e_y)).transpose().conj() @\
        np.hstack((matrix_e_x, matrix_e_y))

    # get eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix_c)
    sorted_index = np.argsort(np.abs(eigenvalues))[::-1]  # descending order
    matrix_e = eigenvectors[:, sorted_index[:2 * num_signal]]

    # take the upper right and lower right sub matrix
    matrix_e_12 = matrix_e[:num_signal, num_signal:]
    matrix_e_22 = matrix_e[num_signal:, num_signal:]

    matrix_psi = -matrix_e_12 @ np.linalg.inv(matrix_e_22)
    matrix_phi = np.linalg.eigvals(matrix_psi)

    # Note: the signal model we use is different from model in reference paper,
    # so there should be "-2 pi f"
    angles = np.arcsin(C * np.angle(matrix_phi) / ((-2 * np.pi * signal_fre) *
                                                   sub_array_spacing))

    if unit == "deg":
        angles = angles / np.pi * 180

    return np.sort(angles)
