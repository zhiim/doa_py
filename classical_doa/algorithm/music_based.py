import numpy as np

from classical_doa.algorithm.utils import get_noise_space

C = 3e8


def music(
    received_data, num_signal, array, signal_fre, angle_grids, unit="deg"
):
    """1D MUSIC

    Args:
        received_data : Array received signals
        num_signal : Number of signals
        array : Instance of array class
        signal_fre: Signal frequency
        angle_grids : Angle grids corresponding to spatial spectrum. It should
            be a numpy array.
        unit : Unit of angle, 'rad' for radians, 'deg' for degrees. Defaults to
            'deg'.
    """
    noise_space = get_noise_space(np.cov(received_data), num_signal)

    # Calculate the manifold matrix when there are incident signal in all
    # grid points
    manifold_all_grids = array.steering_vector(
        signal_fre, angle_grids, unit=unit
    )

    v = noise_space.transpose().conj() @ manifold_all_grids

    # Each column of matrix v corresponds to an incident signal, calculate the
    # square of the 2-norm for each column
    spectrum = 1 / np.linalg.norm(v, axis=0) ** 2

    return np.squeeze(spectrum)


def root_music(received_data, num_signal, array, signal_fre, unit="deg"):
    """Root-MUSIC

    Args:
        received_data : Array of received signals
        num_signal : Number of signals
        array : Instance of array class
        signal_fre: Signal frequency
        unit: The unit of the angle, `rad` represents radian, `deg` represents
            degree. Defaults to 'deg'.

    References:
        Rao, B.D., and K.V.S. Hari. “Performance Analysis of Root-Music.”
        IEEE Transactions on Acoustics, Speech, and Signal Processing 37,
        no. 12 (December 1989): 1939-49. https://doi.org/10.1109/29.45540.
    """
    noise_space = get_noise_space(np.cov(received_data), num_signal)

    num_antennas = array.num_antennas
    antenna_spacing = array.array_position[1][1] - array.array_position[0][1]

    # Since the polynomial solving function provided by numpy requires the
    # coefficients of the polynomial as input, and extracting the coefficients
    # is very complex, so the implementation code of rootMMUSIC in doatools is
    # directly used here.

    # Alternatively, the sympy library can be used to solve polynomial
    # equations, but it will be more computationally expensive.

    # Compute the coefficients for the polynomial.
    matrix_c = noise_space @ noise_space.transpose().conj()
    coeff = np.zeros((num_antennas - 1,), dtype=np.complex128)
    for i in range(1, num_antennas):
        coeff[i - 1] += np.sum(np.diag(matrix_c, i))
    coeff = np.hstack((coeff[::-1], np.sum(np.diag(matrix_c)), coeff.conj()))
    # Find the roots of the polynomial.
    z = np.roots(coeff)

    # To avoid simultaneously obtaining a pair of complex conjugate roots, only
    # take roots inside the unit circle
    roots_inside_unit_circle = np.extract(np.abs(z) <= 1, z)
    sorted_index = np.argsort(np.abs(np.abs(roots_inside_unit_circle) - 1))
    chosen_roots = roots_inside_unit_circle[sorted_index[:num_signal]]

    angles = np.arcsin(
        (C / signal_fre)
        / (-2 * np.pi * antenna_spacing)
        * np.angle(chosen_roots)
    )

    if unit == "deg":
        angles = angles / np.pi * 180

    return np.sort(angles)


def uca_rb_music(
    received_data,
    num_signal,
    array,
    signal_fre,
    azimuth_grids,
    elevation_grids,
    unit="deg",
):
    """form MUSIC for Uniform Circular Array (UCA)

    Args:
        received_data (_type_): _description_
        num_signal (_type_): _description_
        array (_type_): _description_
        signal_fre (_type_): _description_
        angle_grids (_type_): _description_
        unit (str, optional): _description_. Defaults to "deg".

    References:
        Mathews, C.P., and M.D. Zoltowski. “Eigenstructure Techniques for 2-D
        Angle Estimation with Uniform Circular Arrays.” IEEE Transactions on
        Signal Processing 42, no. 9 (September 1994): 2395-2407.
        https://doi.org/10.1109/78.317861.
    """
    # max number of phase modes can be excitated
    m = np.floor(2 * np.pi * array.radius / (C / signal_fre))

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
    noise_space = get_noise_space(cov_real, num_signal)

    spectrum = np.zeros((azimuth_grids.size, elevation_grids.size))
    for i, elevation in enumerate(elevation_grids):
        angle_grids = np.vstack(
            (azimuth_grids, elevation * np.ones_like(azimuth_grids))
        )
        # Calculate the manifold matrix when there are incident signal in all
        # grid points
        manifold_all_grids = array.steering_vector(
            signal_fre, angle_grids, unit=unit
        )
        manifold_all_grids = matrix_f_r.conj().transpose() @ manifold_all_grids

        v = noise_space.transpose().conj() @ manifold_all_grids

        # Each column of matrix v corresponds to an incident signal, calculate
        # the square of the 2-norm for each column
        spectrum[:, i] = 1 / np.linalg.norm(v, axis=0) ** 2

    return spectrum
