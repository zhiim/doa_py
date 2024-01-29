import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classical_doa.algorithm.broadband.imusic import imusic
from classical_doa.arrays import UniformLinearArray
from classical_doa.plot import plot_spatial_spectrum
from classical_doa.signals import ChirpSignal

# 仿真参数
angle_incidence = np.array([0, 50])
num_snapshots = 300
f0 = np.array([2e6, 3e6])
f1 = np.array([5e6, 6e6])
fs = 2e7
snr = 0

num_antennas = 8
antenna_spacing = 0.5 * (3e8 / max(f1))  # 阵元间距半波长

# 生成仿真信号
signal = ChirpSignal(nsamples=num_snapshots, fs=fs, f0=f0, f1=f1)

array = UniformLinearArray(m=num_antennas, dd=antenna_spacing)

received_data = array.received_signal(signal=signal, snr=snr,
                                      angle_incidence=angle_incidence,
                                      broadband=True,
                                      unit="deg")

# 运行算法
search_grids = np.arange(-90, 90, 1)

num_signal = len(angle_incidence)
spectrum = imusic(received_data=received_data, num_signal=num_signal,
                array=array, fs=fs,
                angle_grids=search_grids, num_groups=5, unit="deg")

plot_spatial_spectrum(spectrum=spectrum, ground_truth=angle_incidence,
                      angle_grids=search_grids, peak_threshold=0.5)
