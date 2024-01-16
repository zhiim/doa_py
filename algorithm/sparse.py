import numpy as np

C = 3e8


def omp(received_data, num_signal, array_position, signal_fre, angle_grids,
        unit="deg"):
    """OMP based sparse representation algorithms for DOA estimation

    Args:
        received_data : 阵列接受信号
        num_signal : 信号个数
        array_position : 阵元位置, 应该是numpy array的形式, 行向量列向量均可
        signal_fre: 信号频率
        angle_grids : 空间谱的网格点, 应该是numpy array的形式
        unit : 角度的单位制, `rad`代表弧度制, `deg`代表角度制. Defaults to
            'deg'.

    Reference:
        Cotter, Shane F. “Multiple Snapshot Matching Pursuit for Direction of
        Arrival (DOA) Estimation.” In 2007 15th European Signal Processing
        Conference, 247-51, 2007.
        https://ieeexplore.ieee.org/abstract/document/7098802.
    """
    if unit == "deg":
        angle_grids = angle_grids / 180 * np.pi

    array_position = array_position.reshape(-1, 1)
    angle_grids = angle_grids.reshape(1, -1)

    # build the overcomplete basis
    tau_all_grids = 1 / C * array_position @ np.sin(angle_grids)
    matrix_a_over = np.exp(-1j * 2 * np.pi * signal_fre * tau_all_grids)

    # initiate iteration
    atom_index = []
    residual = received_data

    # iteration
    while len(atom_index) < num_signal:
        # measure relevance using Frobenius norm
        relevance = np.linalg.norm(matrix_a_over.transpose().conj() @ residual,
                                   axis=1)
        index_max = np.argmax(relevance)
        # append index of atoms
        if index_max not in atom_index:
            atom_index.append(index_max)
        # update residual
        chosen_atom = np.asmatrix(matrix_a_over[:, atom_index])
        sparse_vector = np.linalg.inv(
            chosen_atom.transpose().conj() @ chosen_atom
            ) @ chosen_atom.transpose().conj() @ received_data
        residual = received_data - chosen_atom @ sparse_vector

    angles = angle_grids[0][atom_index]

    if unit == "deg":
        angles = angles / np.pi * 180

    return np.sort(angles)
