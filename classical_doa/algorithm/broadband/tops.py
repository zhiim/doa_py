import numpy as np

from classical_doa.algorithm.utils import (
    divide_into_fre_bins,
    get_noise_space,
    get_signal_space,
)

C = 3e8


def tops(received_data, num_signal, array_position, fs, num_groups, angle_grids,
         fre_ref, unit="deg"):
    """Test of orthogonality of projected subspaces (TOPS) method for wideband
    DOA estimation.

    Args:
        received_data : 阵列接受信号
        num_signal : 信号个数
        array_position : 阵元位置, 应该是numpy array的形式, 行向量列向量均可
        fs: 采样频率
        num_groups: FFT的组数, 每一组都独立做FFT
        angle_grids : 空间谱的网格点, 应该是numpy array的形式
        fre_ref: 参考频点
        unit : 角度的单位制, `rad`代表弧度制, `deg`代表角度制. Defaults to
            'deg'.

    References:
        Yoon, Yeo-Sun, L.M. Kaplan, and J.H. McClellan. “TOPS: New DOA Estimator
        for Wideband Signals.” IEEE Transactions on Signal Processing 54, no. 6
        (June 2006): 1977-89. https://doi.org/10.1109/TSP.2006.872581.
    """
    if unit == "deg":
        angle_grids = angle_grids / 180 * np.pi

    num_antennas = received_data.shape[0]
    array_position = array_position.reshape(-1, 1)

    signal_fre_bins, fre_bins = divide_into_fre_bins(received_data, num_groups,
                                                     fs)

    # index of reference frequency if FFT output
    ref_index = int(fre_ref / (fs / fre_bins.size))
    # get signal space of reference frequency
    signal_space_ref = get_signal_space(signal_fre_bins[:, ref_index, :],
                                        num_signal=num_signal)

    spectrum = np.zeros(angle_grids.size)
    for i, grid in enumerate(angle_grids):
        matrix_d = np.empty((num_signal, 0), dtype=np.complex_)

        for j, fre in enumerate(fre_bins):
            # 计算当前频点对应的噪声子空间
            noise_space_f = get_noise_space(signal_fre_bins[:, j, :],
                                            num_signal)

            # 构造变换矩阵
            matrix_phi = np.exp(-1j * 2 * np.pi * (fre - fre_ref) / C *
                                array_position * np.sin(grid))
            matrix_phi = np.diag(np.squeeze(matrix_phi))

            # 使用变换矩阵将参考频点的信号子空间变换到当前频点
            matrix_u = matrix_phi @ signal_space_ref

            # 构造投影矩阵，减小矩阵U中的误差
            matrix_a_f = np.exp(-1j * 2 * np.pi * fre / C *
                                array_position * np.sin(grid))
            matrix_p = np.eye(num_antennas) -\
                1 / (matrix_a_f.transpose().conj() @ matrix_a_f) *\
                    matrix_a_f @ matrix_a_f.transpose().conj()

            # 使用投影矩阵对矩阵U进行投影
            matrix_u = matrix_p @ matrix_u

            matrix_d = np.concatenate((matrix_d,
                                       matrix_u.T.conj() @ noise_space_f),
                                       axis=1)

        # 使用矩阵D中的最小特征值构造空间谱
        _, s, _ = np.linalg.svd(matrix_d)
        spectrum[i] = 1 / min(s)

    return spectrum
