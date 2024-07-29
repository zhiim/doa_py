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
    signal_space = get_signal_space(np.cov(received_data), num_signal)

    # get signal space of two sub array. Each sub array consists of M-1 antennas
    matrix_e_x = signal_space[:-1, :]
    matrix_e_y = signal_space[1:, :]
    # the fixed distance of corresponding elements in two sub-array ensures
    # the rotational invariance
    sub_array_spacing = array.array_position[1][1] - array.array_position[0][1]

    matrix_c = np.hstack(
        (matrix_e_x, matrix_e_y)
    ).transpose().conj() @ np.hstack((matrix_e_x, matrix_e_y))

    # get eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix_c)
    sorted_index = np.argsort(np.abs(eigenvalues))[::-1]  # descending order
    matrix_e = eigenvectors[:, sorted_index[: 2 * num_signal]]

    # take the upper right and lower right sub matrix
    matrix_e_12 = matrix_e[:num_signal, num_signal:]
    matrix_e_22 = matrix_e[num_signal:, num_signal:]

    matrix_psi = -matrix_e_12 @ np.linalg.inv(matrix_e_22)
    matrix_phi = np.linalg.eigvals(matrix_psi)

    # Note: the signal model we use is different from model in reference paper,
    # so there should be "-2 pi f"
    angles = np.arcsin(
        C
        * np.angle(matrix_phi)
        / ((-2 * np.pi * signal_fre) * sub_array_spacing)
    )

    if unit == "deg":
        angles = angles / np.pi * 180

    return np.sort(angles)


def uca_esprit(received_data, num_signal, array, signal_fre, unit="deg"):
    """UCA-ESPRIT for Uniform Circular Array.

    Args:
        received_data : Array received signals
        num_signal : Number of signals
        array : Instance of array class
        signal_fre: Signal frequency
        unit : Unit of angle, 'rad' for radians, 'deg' for degrees. Defaults to
            'deg'.

    Reference:
        Mathews, C.P., and M.D. Zoltowski. “Eigenstructure Techniques for 2-D
        Angle Estimation with Uniform Circular Arrays.” IEEE Transactions on
        Signal Processing 42, no. 9 (September 1994): 2395-2407.
        https://doi.org/10.1109/78.317861.
    """
    # max number of phase modes can be excitated
    m = int(np.floor(2 * np.pi * array.radius / (C / signal_fre)))

    matrix_c_v = np.diag(
        1j ** np.concatenate((np.arange(-m, 0), np.arange(0, -m - 1, step=-1)))
    )
    matrix_v = (
        1
        / np.sqrt(array.num_antennas)
        * np.exp(
            -1j
            * 2
            * np.pi
            * np.arange(0, array.num_antennas).reshape(-1, 1)
            @ np.arange(-m, m + 1).reshape(1, -1)
            / array.num_antennas
        )
    )
    matrix_f_e = matrix_v @ matrix_c_v.conj().transpose()
    matrix_w = (
        1
        / np.sqrt(2 * m + 1)
        * np.exp(
            1j
            * 2
            * np.pi
            * np.arange(-m, m + 1).reshape(-1, 1)
            @ np.arange(-m, m + 1).reshape(1, -1)
            / (2 * m + 1)
        )
    )
    matrix_f_r = matrix_f_e @ matrix_w

    # beamspace data vector
    beamspace_data = matrix_f_r.conj().transpose() @ received_data

    # only use the real part of covariance matrix
    cov_real = np.real(np.cov(beamspace_data))
    signal_space = get_signal_space(cov_real, num_signal)

    matrix_c_o = np.diag(
        (-1) ** np.concatenate((np.arange(m, -1, step=-1), np.zeros(m)))
    )
    signal_space = matrix_c_o @ matrix_w @ signal_space

    s1 = signal_space[:-2, :]
    s2 = signal_space[1:-1, :]
    s3 = signal_space[2:, :]

    matrix_gamma = (1 / np.pi / array.radius) * np.diag(np.arange(-(m - 1), m))
    matrix_e = np.hstack((s1, s3))
    matrix_psi_hat = (
        np.linalg.inv(matrix_e.conj().transpose() @ matrix_e)
        @ matrix_e.conj().transpose()
        @ matrix_gamma
        @ s2
    )
    matrix_psi = matrix_psi_hat[: len(matrix_psi_hat) // 2, :]

    eig_values = np.linalg.eigvals(matrix_psi)

    elevation = np.arccos(np.abs(eig_values))
    azimuth = np.angle(-eig_values)

    if unit == "deg":
        elevation = elevation / np.pi * 180
        azimuth = azimuth / np.pi * 180

    return azimuth, elevation
