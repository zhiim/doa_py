import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
import numpy as np

from doa_py.algorithm import cssm, imusic, tops
from doa_py.arrays import UniformLinearArray
from doa_py.plot import plot_spatial_spectrum
from doa_py.signals import ChirpSignal

# signal parameters
angle_incidence = np.array([0, 30])
num_snapshots = 1000
fre_min = 1e6
fre_max = 1e7
fs = 2.5e7
snr = 0

num_antennas = 8
antenna_spacing = 0.5 * (
    3e8 / fre_max
)  # set to half wavelength of highest frequency


# generate signal and received data
signal = ChirpSignal(f_min=fre_min, f_max=fre_max, fs=fs)

array = UniformLinearArray(m=num_antennas, dd=antenna_spacing)

# plot the signal in the frequency domain
plt.plot(
    np.fft.fftshift(np.fft.fftfreq(num_snapshots, 1 / fs)),
    np.abs(
        np.fft.fftshift(
            np.fft.fft(
                signal.gen(n=len(angle_incidence), nsamples=num_snapshots)
            )
        )
    ).transpose(),
)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()

received_data = array.received_signal(
    signal=signal,
    snr=snr,
    nsamples=num_snapshots,
    angle_incidence=angle_incidence,
    unit="deg",
)


search_grids = np.arange(-90, 90, 1)

num_signal = len(angle_incidence)
spectrum = imusic(
    received_data=received_data,
    num_signal=num_signal,
    array=array,
    fs=fs,
    angle_grids=search_grids,
    num_groups=16,
    unit="deg",
)

plot_spatial_spectrum(
    spectrum=spectrum,
    ground_truth=angle_incidence,
    angle_grids=search_grids,
    num_signal=num_signal,
)

spectrum = cssm(
    received_data=received_data,
    num_signal=num_signal,
    array=array,
    fs=fs,
    angle_grids=search_grids,
    fre_ref=(fre_min + fre_max) / 2,
    pre_estimate=np.array([-1, 29]),
    unit="deg",
)

plot_spatial_spectrum(
    spectrum=spectrum,
    ground_truth=angle_incidence,
    angle_grids=search_grids,
    num_signal=num_signal,
)

spectrum = tops(
    received_data=received_data,
    num_signal=num_signal,
    array=array,
    fs=fs,
    num_groups=32,
    angle_grids=search_grids,
    fre_ref=4e6,
    unit="deg",
)

plot_spatial_spectrum(
    spectrum=spectrum,
    ground_truth=angle_incidence,
    angle_grids=search_grids,
    num_signal=num_signal,
)
