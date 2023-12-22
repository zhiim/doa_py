import numpy as np

def get_narrowband_array_received_data(array, signal, azimuth,
                                       elevation = None, unit = 'rad'):
    """计算阵列接收信号

    Args:
        array: 阵列实例
        signal: 信号实例
        azimuth (ndarray): 信号入射的方位角 (多个信号的入射角度)
        elevation (ndarray): 信号入射的俯仰角 (多个信号的入射角度)
        unit (str): {'rad', 'deg'}, 角度制或弧度制

    Return:
        received_data: M x L 维的矩阵, 其中M是阵元数, L是采样点数
    """
    steering_maxtrix = np.zeros(array.num_elements, signal.num_signals)
    for i in range(signal.num_signals):
        steering_maxtrix[:, i] = array.steering_vector(signal.frequency,
                                                       azimuth[i], elevation[i],
                                                       unit)
    received_data = steering_maxtrix @ signal.gen()
    return received_data
