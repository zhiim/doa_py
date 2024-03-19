import numpy as np
import cvxpy as cp

C = 3e8


def omp(received_data, num_signal, array, signal_fre, angle_grids, unit="deg"):
    """OMP based sparse representation algorithms for DOA estimation

    Args:
        received_data : Array received signals
        num_signal : Number of signals
        array : Instance of array class
        signal_fre: Signal frequency
        angle_grids : Angle grids corresponding to spatial spectrum. It should
            be a numpy array.
        unit : Unit of angle, 'rad' for radians, 'deg' for degrees. Defaults to
            'deg'.

    Reference:
        Cotter, Shane F. “Multiple Snapshot Matching Pursuit for Direction of
        Arrival (DOA) Estimation.” In 2007 15th European Signal Processing
        Conference, 247-51, 2007.
        https://ieeexplore.ieee.org/abstract/document/7098802.
    """
    angle_grids = angle_grids.reshape(
        -1,
    )

    # # build the overcomplete basis
    matrix_a_over = array.steering_vector(signal_fre, angle_grids, unit=unit)

    # initiate iteration
    atom_index = []
    residual = received_data

    # iteration
    while len(atom_index) < num_signal:
        # measure relevance using Frobenius norm
        relevance = np.linalg.norm(
            matrix_a_over.transpose().conj() @ residual, axis=1
        )
        index_max = np.argmax(relevance)
        # append index of atoms
        if index_max not in atom_index:
            atom_index.append(index_max)
        # update residual
        chosen_atom = np.asmatrix(matrix_a_over[:, atom_index])
        sparse_vector = (
            np.linalg.inv(chosen_atom.transpose().conj() @ chosen_atom)
            @ chosen_atom.transpose().conj()
            @ received_data
        )
        residual = received_data - chosen_atom @ sparse_vector

    angles = angle_grids[atom_index]

    return np.sort(angles)


def l1_svd(
    received_data, num_signal, array, signal_fre, angle_grids, unit="deg"
):
    """L1 norm based sparse representation algorithms for DOA estimation

    Args:
        received_data : Array received signals
        num_signal : Number of signals
        array : Instance of array class
        signal_fre: Signal frequency
        angle_grids : Angle grids corresponding to spatial spectrum. It should
            be a numpy array.
        unit : Unit of angle, 'rad' for radians, 'deg' for degrees. Defaults to
            'deg'.

    Reference:
        Malioutov, D., M. Cetin, and A.S. Willsky. “A Sparse Signal
        Reconstruction Perspective for Source Localization with Sensor Arrays.”
        IEEE Transactions on Signal Processing 53, no. 8 (August 2005): 3010-22.
        https://doi.org/10.1109/TSP.2005.850882.
    """
    # build the overcomplete basis
    a_over = array.steering_vector(signal_fre, angle_grids, unit=unit)

    num_samples = received_data.shape[1]

    _, _, vh = np.linalg.svd(received_data)

    d_k = np.vstack(
        (np.eye(num_signal), np.zeros((num_samples - num_signal, num_signal)))
    )
    y_sv = received_data @ vh.conj().transpose() @ d_k

    # solve the l1 norm problem using cvxpy
    p = cp.Variable()
    q = cp.Variable()
    r = cp.Variable(len(angle_grids))
    s_sv = cp.Variable((len(angle_grids), num_signal), complex=True)

    # constraints of the problem
    constraints = [cp.norm(y_sv - a_over @ s_sv, "fro") <= p, cp.sum(r) <= q]
    for i in range(len(angle_grids)):
        constraints.append(cp.norm(s_sv[i, :]) <= r[i])

    # objective function
    objective = cp.Minimize(p + 2 * q)
    prob = cp.Problem(objective, constraints)

    prob.solve()

    spectrum = s_sv.value
    spectrum = np.sum(np.abs(spectrum), axis=1)

    return spectrum
