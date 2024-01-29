import numpy as np

from classical_doa.algorithm.music import music
from classical_doa.algorithm.utils import divide_into_fre_bins


def imusic(received_data, num_signal, array, fs, angle_grids,
           num_groups, unit="deg"):
    """Incoherent MUSIC estimator for wideband DOA estimation.

    Args:
        received_data : Array received signals
        num_signal : Number of signals
        array : Instance of array class
        fs: sampling frequency
        angle_grids : Angle grids corresponding to spatial spectrum. It should
            be a numpy array.
        num_groups: Divide sampling points into serveral groups, and do FFT
            separately in each group
        unit : Unit of angle, 'rad' for radians, 'deg' for degrees. Defaults to
            'deg'.

    References:
        Wax, M., Tie-Jun Shan, and T. Kailath. “Spatio-Temporal Spectral
        Analysis by Eigenstructure Methods.” IEEE Transactions on Acoustics,
        Speech, and Signal Processing 32, no. 4 (August 1984): 817-27.
        https://doi.org/10.1109/TASSP.1984.1164400.
    """
    signal_fre_bins, fre_bins = divide_into_fre_bins(received_data, num_groups,
                                                     fs)

    # MUSIC algorithm in every frequency point
    spectrum_fre_bins = np.zeros((signal_fre_bins.shape[1], angle_grids.size))
    for i, fre in enumerate(fre_bins):
        spectrum_fre_bins[i, :] = music(received_data=signal_fre_bins[:, i, :],
                                        num_signal=num_signal,
                                        array=array,
                                        signal_fre=fre,
                                        angle_grids=angle_grids,
                                        unit=unit)

    spectrum = np.mean(spectrum_fre_bins, axis=0)

    return np.squeeze(spectrum)
