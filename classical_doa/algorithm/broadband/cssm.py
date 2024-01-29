import numpy as np

from classical_doa.algorithm.music import music

C = 3e8


def cssm(received_data, num_signal, array, fs, angle_grids, fre_ref,
         pre_estimate, unit="deg"):
    """Coherent Signal Subspace Method (CSSM) for wideband DOA estimation.

    Args:
        received_data : Array received signals
        num_signal : Number of signals
        array : Instance of array class
        fs: sampling frequency
        angle_grids : Angle grids corresponding to spatial spectrum. It should
            be a numpy array.
        fre_ref: reference frequency
        pre_estimate: pre-estimated angles
        unit : Unit of angle, 'rad' for radians, 'deg' for degrees. Defaults to
            'deg'.

    References:
        Wang, H., and M. Kaveh. “Coherent Signal-Subspace Processing for the
        Detection and Estimation of Angles of Arrival of Multiple Wide-Band
        Sources.” IEEE Transactions on Acoustics, Speech, and Signal Processing
        33, no. 4 (August 1985): 823-31.
        https://doi.org/10.1109/TASSP.1985.1164667.
    """
    num_snapshots = received_data.shape[1]
    pre_estimate = pre_estimate.reshape(1, -1)

    # Divide the received signal into multiple frequency points
    signal_fre_bins = np.fft.fft(received_data, axis=1)
    fre_bins = np.fft.fftfreq(num_snapshots, 1 / fs)

    # Calculate the manifold matrix corresponding to the pre-estimated angles at
    # the reference frequency point
    matrix_a_ref = array.steering_vector(fre_ref, pre_estimate, unit=unit)

    for i, fre in enumerate(fre_bins):
        # Manifold matrix corresponding to the pre-estimated angles at
        # each frequency point
        matrix_a_f = array.steering_vector(fre, pre_estimate, unit=unit)
        matrix_q = matrix_a_f @ matrix_a_ref.transpose().conj()
        # Perform singular value decomposition on matrix_q
        matrix_u, _, matrix_vh = np.linalg.svd(matrix_q)
        # Construct the optimal focusing matrix using the RSS method
        matrix_t_f = matrix_vh.transpose().conj() @ matrix_u.transpose().conj()
        # Focus the received signals at each frequency point to the reference
        # frequency point
        signal_fre_bins[:, i] = matrix_t_f @ signal_fre_bins[:, i]

    spectrum = music(received_data=signal_fre_bins, num_signal=num_signal,
                     array=array, signal_fre=fre_ref,
                     angle_grids=angle_grids, unit=unit)

    return np.squeeze(spectrum)
