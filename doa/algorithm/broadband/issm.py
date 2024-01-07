import numpy as np
from doa.algorithm.music import music

def issm(received_data, num_signal, array_position, fs, angle_grids, num_groups,
         unit="deg"):
    num_snapshots = received_data.shape[1]

    n_each_group = num_snapshots // num_groups
    if n_each_group < 128:
        n_fft = 128
    else:
        n_fft = n_each_group

    signal_fre_bins = np.zeros((received_data.shape[0], n_fft, num_groups),
                               dtype=np.complex_)
    for group_i in range(num_groups):
        signal_fre_bins[:, :, group_i] = np.fft.fft(
            received_data[:,
                          group_i * n_each_group: (group_i + 1) * n_each_group],
            n=n_fft,
            axis=1
            )
    fre_bins = np.fft.fftfreq(n_fft, 1 / fs)

    spectrum_fre_bins = np.zeros((n_fft, angle_grids.size))
    for i, fre in enumerate(fre_bins):
        spectrum_fre_bins[i, :] = music(received_data=signal_fre_bins[:, i, :],
                                        num_signal=num_signal,
                                        array_position=array_position,
                                        signal_fre=fre,
                                        angle_grids=angle_grids,
                                        unit=unit)

    spectrum = np.mean(spectrum_fre_bins, axis=0)

    return np.squeeze(spectrum)
