import numpy as np

from classical_doa.algorithm.utils import get_noise_space

C = 3e8


def music(received_data, num_signal, array_position, signal_fre, angle_grids,
          unit="deg"):
    """1D MUSIC

    Args:
        received_data : 阵列接受信号
        num_signal : 信号个数
        array_position : 阵元位置, 应该是numpy array的形式, 行向量列向量均可
        signal_fre: 信号频率
        angle_grids : 空间谱的网格点, 应该是numpy array的形式
        unit : 角度的单位制, `rad`代表弧度制, `deg`代表角度制. Defaults to
            'deg'.
    """
    noise_space = get_noise_space(received_data, num_signal)

    # 变为列向量用于后续矩阵计算
    array_position = np.reshape(array_position, (-1, 1))

    if unit == "deg":
        angle_grids = angle_grids / 180 * np.pi
    angle_grids = np.reshape(angle_grids, (1, -1))

    # 计算所有网格点信号入射时的流型矩阵
    tau_all_grids = 1 / C * array_position @ np.sin(angle_grids)
    manifold_all_grids = np.exp(-1j * 2 * np.pi * signal_fre * tau_all_grids)

    v = noise_space.transpose().conj() @ manifold_all_grids

    # 矩阵v的每一列对应一个入射信号, 对每一列求二范数的平方
    spectrum = 1 / np.linalg.norm(v, axis=0) ** 2

    return np.squeeze(spectrum)

def root_music(received_data, num_signal, array_position, signal_fre,
               unit="deg"):
    """Root-MUSIC

    Args:
        received_data : 阵列接受信号
        num_signal : 信号个数
        array_position : 阵元位置, 应该是numpy array的形式, 行向量列向量均可
        signal_fre: 信号频率
        unit : 角度的单位制, `rad`代表弧度制, `deg`代表角度制. Defaults to
            'deg'.

    References:
        Rao, B.D., and K.V.S. Hari. “Performance Analysis of Root-Music.”
        IEEE Transactions on Acoustics, Speech, and Signal Processing 37,
        no. 12 (December 1989): 1939-49. https://doi.org/10.1109/29.45540.
    """
    noise_space = get_noise_space(received_data, num_signal)

    num_antennas = array_position.size
    antenna_spacing = array_position[1] - array_position[0]

    # 由于numpy提供的多项式求解函数需要以多项式的系数作为输入，而系数的提取非常
    # 复杂, 此处直接搬用doatools中rootMMUSIC的实现代码
    # 也可以使用sympy库求解多项式方程，但是计算量会更大
    # Compute the coefficients for the polynomial.
    matrix_c = noise_space @ noise_space.transpose().conj()
    coeff = np.zeros((num_antennas - 1,), dtype=np.complex_)
    for i in range(1, num_antennas):
        coeff[i - 1] += np.sum(np.diag(matrix_c, i))
    coeff = np.hstack((coeff[::-1], np.sum(np.diag(matrix_c)), coeff.conj()))
    # Find the roots of the polynomial.
    z = np.roots(coeff)

    # 为了防止同时取到一对共轭根, 只取单位圆内的根
    # find k roots inside and closest to the unit circle
    roots_inside_unit_circle = np.extract(np.abs(z) <= 1, z)
    sorted_index = np.argsort(np.abs(np.abs(roots_inside_unit_circle) - 1))
    chosen_roots = roots_inside_unit_circle[sorted_index[:num_signal]]

    angles = np.arcsin((C / signal_fre) / (-2 * np.pi * antenna_spacing) *
                       np.angle(chosen_roots))

    if unit == "deg":
        angles = angles / np.pi * 180

    return np.sort(angles)
