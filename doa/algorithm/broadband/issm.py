import numpy as np
from doa.algorithm.music import music

def issm(received_data, num_signal, array_position, fs, angle_grids, num_groups,
         unit="deg"):
    """Incoherent Signal Subspace Method (ISSM) estimator for wideband DOA esti-
    mation.

    Args:
        received_data : 阵列接受信号
        num_signal : 信号个数
        array_position : 阵元位置, 应该是numpy array的形式, 行向量列向量均可
        fs: 采样频率
        angle_grids : 空间谱的网格点, 应该是numpy array的形式
        num_groups: FFT的组数, 每一组都独立做FFT
        unit : 角度的单位制, `rad`代表弧度制, `deg`代表角度制. Defaults to
            'deg'.

    References:
        Wax, M., Tie-Jun Shan, and T. Kailath. “Spatio-Temporal Spectral
        Analysis by Eigenstructure Methods.” IEEE Transactions on Acoustics,
        Speech, and Signal Processing 32, no. 4 (August 1984): 817-27.
        https://doi.org/10.1109/TASSP.1984.1164400.
    """
    num_snapshots = received_data.shape[1]

    n_each_group = num_snapshots // num_groups  # 每一组包含的采样点数
    if n_each_group < 128:
        n_fft = 128  # 如果点数太少, 做FFT时补零
    else:
        n_fft = n_each_group

    signal_fre_bins = np.zeros((received_data.shape[0], n_fft, num_groups),
                               dtype=np.complex_)
    # 每一组独立做FFT
    for group_i in range(num_groups):
        signal_fre_bins[:, :, group_i] = np.fft.fft(
            received_data[:,group_i * n_each_group: (group_i+1) * n_each_group],
            n=n_fft,
            axis=1
            )
    fre_bins = np.fft.fftfreq(n_fft, 1 / fs)

    # 对每一个频点运行MUSIC算法
    spectrum_fre_bins = np.zeros((n_fft, angle_grids.size))
    for i, fre in enumerate(fre_bins):
        spectrum_fre_bins[i, :] = music(received_data=signal_fre_bins[:, i, :],
                                        num_signal=num_signal,
                                        array_position=array_position,
                                        signal_fre=fre,
                                        angle_grids=angle_grids,
                                        unit=unit)

    # 取平均
    spectrum = np.mean(spectrum_fre_bins, axis=0)

    return np.squeeze(spectrum)
