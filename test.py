import os
import sys
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doa.signal import ComplexStochasticSignal
from doa.array import UniformLinearArray


C = 3e8

# 仿真参数
angle_incidence = np.array([-60, 0, 30])
num_signal = len(angle_incidence)
num_snapshots = 1
signal_fre = 2e7
fs = 5e7
snr = 20

num_antennas = 8
antenna_spacing = 0.5 * (3e8 / signal_fre)  # 阵元间距半波长

# 生成仿真信号
signal = ComplexStochasticSignal(n=num_signal, nsamples=num_snapshots,
                                 fre=signal_fre, fs=fs)

array = UniformLinearArray(m=num_antennas, dd=antenna_spacing)

received_data = array.received_signal(signal=signal, snr=snr,
                                      angle_incidence=angle_incidence,
                                      unit="deg")

# 运行MUSIC算法
search_grids = np.arange(-90, 90, 1)

element_positon = np.arange(num_antennas).reshape(-1, 1) * antenna_spacing
matrix_tau = 1 / C * element_positon @ np.sin(search_grids.reshape(1, -1))
manifold_over = np.exp(-1j * 2 * np.pi * signal_fre * matrix_tau)

# epsilon =  1/(10**(snr/20)) * np.sqrt(num_snapshots) * np.sqrt(1 + 2*np.sqrt(2)/np.sqrt(num_snapshots));
epsilon = 0.8

x = cp.Variable((len(search_grids)), complex=complex)

# obj_func = 0.5 * cp.norm(cp.matmul(manifold_over, x) - received_data, 'fro')**2\
#     + epsilon * cp.sum(cp.norm(x.T, 2, axis=0))
obj_func = cp.norm(received_data - manifold_over @ x.reshape((-1, 1)), 'fro')**2 + epsilon * cp.norm(x, 1)

problem = cp.Problem(cp.Minimize(obj_func))
result = problem.solve()

print(x.value.shape)
plt.plot(search_grids, np.abs(x.value))
# plt.plot(search_grids, np.abs(np.squeeze(np.linalg.norm(x.value, 2, axis=1))))
plt.show()
