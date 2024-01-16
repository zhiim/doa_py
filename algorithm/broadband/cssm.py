import numpy as np

from classical_doa.algorithm.music import music

C = 3e8


def cssm(received_data, num_signal, array_position, fs, angle_grids, fre_ref,
         pre_estimate, unit="deg"):
    """Coherent Signal Subspace Method (CSSM) for wideband DOA estimation.

    Args:
        received_data : 阵列接受信号
        num_signal : 信号个数
        array_position : 阵元位置, 应该是numpy array的形式, 行向量列向量均可
        fs: 采样频率
        angle_grids : 空间谱的网格点, 应该是numpy array的形式
        fre_ref: 参考频点
        pre_estimate: 角度预估计值
        unit : 角度的单位制, `rad`代表弧度制, `deg`代表角度制. Defaults to
            'deg'.

    References:
        Wang, H., and M. Kaveh. “Coherent Signal-Subspace Processing for the
        Detection and Estimation of Angles of Arrival of Multiple Wide-Band
        Sources.” IEEE Transactions on Acoustics, Speech, and Signal Processing
        33, no. 4 (August 1985): 823-31.
        https://doi.org/10.1109/TASSP.1985.1164667.
    """
    if unit == "deg":
        pre_estimate = pre_estimate / 180 * np.pi

    num_snapshots = received_data.shape[1]
    pre_estimate = pre_estimate.reshape(1, -1)
    array_position = array_position.reshape(-1, 1)

    # 频域下的阵列接收信号
    signal_fre_bins = np.fft.fft(received_data, axis=1)
    fre_bins = np.fft.fftfreq(num_snapshots, 1 / fs)

    # 计算参考频点下，预估计角度对应的流型矩阵
    matrix_a_ref = np.exp(-1j * 2 * np.pi * fre_ref / C *\
                          array_position @ pre_estimate)

    for i, fre in enumerate(fre_bins):
        # 每个频点下，角度预估值对应的流型矩阵
        matrix_a_f = np.exp(-1j * 2 * np.pi * fre / C *\
                            array_position @ np.sin(pre_estimate))
        matrix_q = matrix_a_f @ matrix_a_ref.transpose().conj()
        # 对matrix_q进行奇异值分解
        matrix_u, _, matrix_vh = np.linalg.svd(matrix_q)
        # RSS法构造最佳聚焦矩阵
        matrix_t_f = matrix_vh.transpose().conj() @ matrix_u.transpose().conj()
        # 将每个频点对应的接收信号聚焦到参考频点
        signal_fre_bins[:, i] = matrix_t_f @ signal_fre_bins[:, i]

    spectrum = music(received_data=signal_fre_bins, num_signal=num_signal,
                     array_position=array_position, signal_fre=fre_ref,
                     angle_grids=angle_grids, unit=unit)

    return np.squeeze(spectrum)
