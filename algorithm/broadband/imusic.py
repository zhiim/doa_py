import numpy as np

from classical_doa.algorithm.music import music
from classical_doa.algorithm.utils import divide_into_fre_bins


def imusic(received_data, num_signal, array_position, fs, angle_grids,
           num_groups, unit="deg"):
    """Incoherent MUSIC estimator for wideband DOA estimation.

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
    signal_fre_bins, fre_bins = divide_into_fre_bins(received_data, num_groups,
                                                     fs)

    # 对每一个频点运行MUSIC算法
    spectrum_fre_bins = np.zeros((signal_fre_bins.shape[1], angle_grids.size))
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
