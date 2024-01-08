import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doa.algorithm.music import root_music
from doa.array import UniformLinearArray
from doa.signal import ComplexStochasticSignal

# 仿真参数
angle_incidence = np.array([-60, 0, 30])
num_signal = len(angle_incidence)
num_snapshots = 300
signal_fre = 2e7
fs = 5e7
snr = 0

num_antennas = 8
antenna_spacing = 0.5 * (3e8 / signal_fre)  # 阵元间距半波长

# 生成仿真信号
signal = ComplexStochasticSignal(n=num_signal, nsamples=num_snapshots,
                                 fre=signal_fre, fs=fs)

array = UniformLinearArray(m=num_antennas, dd=antenna_spacing)

received_data = array.received_signal(signal=signal, snr=snr,
                                      angle_incidence=angle_incidence,
                                      unit="deg")

# 运行算法
angles = root_music(received_data=received_data, num_signal=num_signal,
                    array_position=np.arange(8) * antenna_spacing,
                    signal_fre=signal_fre, unit="deg")

np.set_printoptions(suppress=True)
print(angles)
