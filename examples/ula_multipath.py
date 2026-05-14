import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

from doa_py.algorithm import music
from doa_py.arrays import UniformLinearArray
from doa_py.plot import plot_spatial_spectrum
from doa_py.signals import RandomFreqSignal

# signal parameters
num_snapshots = 300
signal_fre = 2e7
fs = 5e7
snr = 0

# array parameters
num_antennas = 8
antenna_spacing = 0.5 * (
    3e8 / signal_fre
)  # set array spacing to half wavelength

# incident angles
angle_incidence = np.array([0, 30])

num_multipath = 2

num_signal = len(angle_incidence) * (1 + num_multipath)

# initialize signal instance
signal = RandomFreqSignal(fc=signal_fre)
signal_ideal = RandomFreqSignal(fc=signal_fre)

# initialize array instance
array = UniformLinearArray(m=num_antennas, dd=antenna_spacing)

signal.add_multipath(2)

# generate received data
received_data = array.received_signal(
    signal=signal,
    snr=snr,
    nsamples=num_snapshots,
    angle_incidence=angle_incidence,
    unit="deg",
)
received_data_ideal = array.received_signal(
    signal=signal_ideal,
    snr=snr,
    nsamples=num_snapshots,
    angle_incidence=angle_incidence,
    unit="deg",
)

search_grids = np.arange(-90, 90, 1)

music_spectrum_ideal = music(
    received_data=received_data_ideal,
    num_signal=len(angle_incidence),
    array=array,
    signal_fre=signal_fre,
    angle_grids=search_grids,
    unit="deg",
)

# plot spatial spectrum
plot_spatial_spectrum(
    spectrum=music_spectrum_ideal,
    angle_grids=search_grids,
    ground_truth=angle_incidence,
    num_signal=len(angle_incidence),
    y_label="MUSIC Spectrum (dB)",
)
plt.savefig("ula_multipath_ideal.svg")

music_spectrum = music(
    received_data=received_data,
    num_signal=num_signal,
    array=array,
    signal_fre=signal_fre,
    angle_grids=search_grids,
    unit="deg",
)

# plot spatial spectrum
plot_spatial_spectrum(
    spectrum=music_spectrum,
    angle_grids=search_grids,
    ground_truth=signal.doa / np.pi * 180,
    num_signal=num_signal,
    y_label="MUSIC Spectrum (dB)",
)
plt.savefig("ula_multipath.svg")
