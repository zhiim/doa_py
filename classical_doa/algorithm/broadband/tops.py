import numpy as np

from classical_doa.algorithm.utils import (
    divide_into_fre_bins,
    get_noise_space,
    get_signal_space,
)

C = 3e8


def tops(received_data, num_signal, array, fs, num_groups, angle_grids,
         fre_ref, unit="deg"):
    """Test of orthogonality of projected subspaces (TOPS) method for wideband
    DOA estimation.

    Args:
        received_data: received signals from the array.
        num_signal: Number of signals.
        array : Instance of array class
        fs: Sampling frequency.
        num_groups: Number of groups for FFT, each group performs an
            independent FFT.
        angle_grids: Grid points of spatial spectrum, should be a numpy array.
        fre_ref: Reference frequency point.
        unit: Unit of angle measurement, 'rad' for radians, 'deg' for degrees.
            Defaults to 'deg'.

    References:
        Yoon, Yeo-Sun, L.M. Kaplan, and J.H. McClellan. “TOPS: New DOA Estimator
        for Wideband Signals.” IEEE Transactions on Signal Processing 54, no. 6
        (June 2006): 1977-89. https://doi.org/10.1109/TSP.2006.872581.
    """
    num_antennas = received_data.shape[0]

    signal_fre_bins, fre_bins = divide_into_fre_bins(received_data, num_groups,
                                                     fs)

    # index of reference frequency in FFT output
    ref_index = int(fre_ref / (fs / fre_bins.size))
    # get signal space of reference frequency
    signal_space_ref = get_signal_space(
        np.cov(signal_fre_bins[:, ref_index, :]),
        num_signal=num_signal
        )

    spectrum = np.zeros(angle_grids.size)
    for i, grid in enumerate(angle_grids):
        matrix_d = np.empty((num_signal, 0), dtype=np.complex128)

        for j, fre in enumerate(fre_bins):
            # calculate noise subspace for the current frequency point
            noise_space_f = get_noise_space(np.cov(signal_fre_bins[:, j, :]),
                                            num_signal)

            # construct transformation matrix
            matrix_phi = array.steering_vector(fre - fre_ref, grid, unit=unit)
            matrix_phi = np.diag(np.squeeze(matrix_phi))

            # transform the signal subspace of the reference frequency to the
            # current frequency using the transformation matrix
            matrix_u = matrix_phi @ signal_space_ref

            # construct projection matrix to reduce errors in matrix U
            matrix_a_f = array.steering_vector(fre, grid, unit=unit)
            matrix_p = np.eye(num_antennas) -\
                1 / (matrix_a_f.transpose().conj() @ matrix_a_f) *\
                    matrix_a_f @ matrix_a_f.transpose().conj()

            # project matrix U using the projection matrix
            matrix_u = matrix_p @ matrix_u

            matrix_d = np.concatenate((matrix_d,
                                       matrix_u.T.conj() @ noise_space_f),
                                       axis=1)

        # construct spatial spectrum using the minimum eigenvalue of matrix D
        _, s, _ = np.linalg.svd(matrix_d)
        spectrum[i] = 1 / min(s)

    return spectrum
