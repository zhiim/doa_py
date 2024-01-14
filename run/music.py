import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classical_doa.algorithm.music import music
from classical_doa.array import UniformLinearArray
from classical_doa.signal import ComplexStochasticSignal

# 仿真参数
angle_incidence = np.array([-60, 0, 30])
num_snapshots = 300
signal_fre = 2e7
fs = 5e7
snr = 0

num_antennas = 8
antenna_spacing = 0.5 * (3e8 / signal_fre)  # 阵元间距半波长

# 生成仿真信号
signal = ComplexStochasticSignal(nsamples=num_snapshots,
                                 fre=signal_fre, fs=fs)

array = UniformLinearArray(m=num_antennas, dd=antenna_spacing)

received_data = array.received_signal(signal=signal, snr=snr,
                                      angle_incidence=angle_incidence,
                                      unit="deg")

# 运行算法
search_grids = np.arange(-90, 90, 1)

num_signal = len(angle_incidence)
spectrum = music(received_data=received_data, num_signal=num_signal,
                 array_position=np.arange(8) * antenna_spacing,
                 signal_fre=signal_fre, angle_grids=search_grids, unit="deg")

plt.plot(search_grids, spectrum)
plt.show()
