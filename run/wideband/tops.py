import sys
sys.path.append('../../')

import numpy as np
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.estimation.grid import FarField1DSearchGrid
from doatools.model.signals import PeriodicChirpSignal
from doatools.estimation.wideband.tops import TOPS
from doatools.model.snapshots import get_wideband_snapshots
from doatools.plotting import plot_spectrum

theta = np.array([-30, 39]) * (np.pi / 180)  # 入射信号的方位角
num_source = len(theta)
f0 = (6e6, 10e6)  # chirp 信号的起始频率
f1 = (14e6, 14e6)  # chirp 信号的终止频率
t1 = (1e-4, 1e-4)  # chirp 信号从f0变化到f1的时间间隔

snr = 5  # 信噪比
s_period = max(t1)  # 采样时长

f_start = min(f0)
f_end = max(f1)
# f_start = 5e6
# f_end = 16e6
fs = 2 * f_end  # 采样频率
d0 = 3e8 / f_end / 2  # 阵元间距

pre_estimate = np.array([-29.1, 39.6]) * (np.pi / 180)

n_fft = 512  # issm算法 进行FFT的点数

ula = UniformLinearArray(n=8, d0=d0)  # 6阵元的均匀线阵
sources = FarField1DSourcePlacement(theta)  # 阵元位置（入射方位）
pcs = PeriodicChirpSignal(dim=num_source, f0=f0, f1=f1, t1=t1,
                          s_period=s_period, fs=fs)  # chirp 信号
# 生成阵元接受到的远场信号
received = get_wideband_snapshots(array=ula, source=sources, source_signal=pcs,
                                  add_noise=True, snr=snr)

grid = FarField1DSearchGrid()  # 算法计算空间谱的grid
estimator = TOPS(array=ula, search_grid=grid)  # tops 算法

resolved, estimates, sp = estimator.estimate(received, fs=fs, f_start=f_start,
                                              f_end=f_end,
                                              n_fft=n_fft,
                                              k=num_source)
if resolved:
    print('Estimates: {0}'.format(estimates.locations))
    print('Ground truth: {0}'.format(sources.locations))
    plot_spectrum({'ISSM': np.abs(sp)}, grid, estimates=estimates,
                        ground_truth=sources, use_log_scale=True,
                        plot_in_deg=True)
