import numpy as np

from classical_doa.algorithm.utils import get_signal_space

C = 3e8


def esprit(received_data, num_signal, array_position, signal_fre, unit="deg"):
    """Total least-squares ESPRIT. Most names of matrix are taken directly from
    the reference paper.

    Args:
        received_data : 阵列接受信号
        num_signal : 信号个数
        array_position : 阵元位置, 应该是numpy array的形式, 行向量列向量均可
        signal_fre: 信号频率
        unit : 返回的估计角度的单位制, `rad`代表弧度制, `deg`代表角度制.
            Defaults to 'deg'.

    Reference:
        Roy, R., and T. Kailath. “ESPRIT-Estimation of Signal Parameters via
        Rotational Invariance Techniques.” IEEE Transactions on Acoustics,
        Speech, and Signal Processing 37, no. 7 (July 1989): 984-95.
        https://doi.org/10.1109/29.32276.
    """
    signal_space = get_signal_space(received_data, num_signal)

    # get signal space of two sub array. Each sub array consists of M-1 antennas
    matrix_e_x = signal_space[:-1, :]
    matrix_e_y = signal_space[1:, :]
    # 两个子阵列中对应阵元之间的固定间距确保了旋转不变性
    sub_array_spacing = array_position[1] - array_position[0]

    matrix_c = np.hstack((matrix_e_x, matrix_e_y)).transpose().conj() @\
        np.hstack((matrix_e_x, matrix_e_y))

    # get eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix_c)
    sorted_index = np.argsort(np.abs(eigenvalues))[::-1]  # 由大到小排序的索引
    matrix_e = eigenvectors[:, sorted_index[:2 * num_signal]]

    # take the upper right and lower right sub matrix
    matrix_e_12 = matrix_e[:num_signal, num_signal:]
    matrix_e_22 = matrix_e[num_signal:, num_signal:]

    matrix_psi = -matrix_e_12 @ np.linalg.inv(matrix_e_22)
    matrix_phi = np.linalg.eigvals(matrix_psi)

    # Note: the signal model we use is different from model in reference paper,
    # so there should be "-2 pi f"
    angles = np.arcsin(C * np.angle(matrix_phi) / ((-2 * np.pi * signal_fre) *
                                                   sub_array_spacing))

    if unit == "deg":
        angles = angles / np.pi * 180

    return np.sort(angles)
