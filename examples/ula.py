import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from doa_py.algorithm import esprit, l1_svd, music, omp, root_music
from doa_py.arrays import UniformLinearArray
from doa_py.plot import plot_estimated_value, plot_spatial_spectrum
from doa_py.signals import ComplexStochasticSignal

# signal parameters
num_snapshots = 300
signal_fre = 2e7
fs = 5e7
snr = -5

# array parameters
num_antennas = 8
antenna_spacing = 0.5 * (
    3e8 / signal_fre
)  # set array spacing to half wavelength

# incident angles
angle_incidence = np.array([0, 30])
num_signal = len(angle_incidence)

# initialize signal instance
signal = ComplexStochasticSignal(fc=signal_fre)

# initialize array instance
array = UniformLinearArray(m=num_antennas, dd=antenna_spacing)

# generate received data
received_data = array.received_signal(
    signal=signal,
    snr=snr,
    nsamples=num_snapshots,
    angle_incidence=angle_incidence,
    unit="deg",
)

search_grids = np.arange(-90, 90, 1)

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

rmusic_estimates = root_music(
    received_data=received_data,
    num_signal=num_signal,
    array=array,
    signal_fre=signal_fre,
    unit="deg",
)
plot_estimated_value(
    estimates=rmusic_estimates,
    ground_truth=angle_incidence,
    y_label="Root-MUSIC Estimated Angle (deg)",
)

esprit_estimates = esprit(
    received_data=received_data,
    num_signal=num_signal,
    array=array,
    signal_fre=signal_fre,
)

plot_estimated_value(
    estimates=esprit_estimates,
    ground_truth=angle_incidence,
    y_label="ESPRIT Estimated Angle (deg)",
)

omp_estimates = omp(
    received_data=received_data,
    num_signal=num_signal,
    array=array,
    signal_fre=signal_fre,
    angle_grids=search_grids,
    unit="deg",
)

plot_estimated_value(
    estimates=omp_estimates,
    ground_truth=angle_incidence,
    y_label="OMP Estimated Angle (deg)",
)

l1_svd_spectrum = l1_svd(
    received_data=received_data,
    num_signal=num_signal,
    array=array,
    signal_fre=signal_fre,
    angle_grids=search_grids,
    unit="deg",
)

plot_spatial_spectrum(
    spectrum=l1_svd_spectrum,
    angle_grids=search_grids,
    ground_truth=angle_incidence,
    num_signal=num_signal,
    y_label="L1-SVD Spectrum (dB)",
)
