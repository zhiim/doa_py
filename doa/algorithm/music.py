import numpy as np

def get_noise_space(received_data, num_signal):
    num_snapshots = received_data.shape[1]

    # compute corvariance matrix
    corvariance_matrix = 1 / num_snapshots *\
        (received_data @ received_data.transpose().conj())

    eigenvalues, eigenvectors = np.linalg.eig(corvariance_matrix)
    sorted_index = np.argsort(np.abs(eigenvalues))  # 由小到大排序的索引
    noise_space = eigenvectors[:, sorted_index[:-num_signal]]

    return noise_space


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
    tau_all_grids = 1 / 3e8 * array_position @ np.sin(angle_grids)
    manifold_all_grids = np.exp(-1j * 2 * np.pi * signal_fre * tau_all_grids)

    v = noise_space.transpose().conj() @ manifold_all_grids

    # 矩阵v的每一列对应一个入射信号, 对每一列求二范数的平方
    spectrum = 1 / np.linalg.norm(v, axis=0) ** 2
    spectrum = 10 * np.log10(spectrum)

    return np.squeeze(spectrum)
