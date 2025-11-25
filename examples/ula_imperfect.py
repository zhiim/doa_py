import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from doa_py.algorithm import music
from doa_py.arrays import UniformLinearArray
from doa_py.plot import plot_spatial_spectrum
from doa_py.signals import ComplexStochasticSignal

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
angle_incidence = np.array([-60, -45, -15, 0, 30])
num_signal = len(angle_incidence)

# initialize signal instance
signal = ComplexStochasticSignal(fc=signal_fre)

# initialize array instance
array = UniformLinearArray(m=num_antennas, dd=antenna_spacing)
array_ideal = UniformLinearArray(m=num_antennas, dd=antenna_spacing)

# add array position imperfections
array.add_position_error_default()

# generate received data
received_data = array.received_signal(
    signal=signal,
    snr=snr,
    nsamples=num_snapshots,
    angle_incidence=angle_incidence,
    unit="deg",
)
received_data_ideal = array_ideal.received_signal(
    signal=signal,
    snr=snr,
    nsamples=num_snapshots,
    angle_incidence=angle_incidence,
    unit="deg",
)

search_grids = np.arange(-90, 90, 1)

music_spectrum_ideal = music(
    received_data=received_data_ideal,
    num_signal=num_signal,
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
    num_signal=num_signal,
    y_label="MUSIC Spectrum (dB)",
)

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
    ground_truth=angle_incidence,
    num_signal=num_signal,
    y_label="MUSIC Spectrum (dB)",
)
