import numpy as np

def get_narrowband_array_received_data(array, signal, azimuth,
                                       elevation = None):
    """计算整理接收信号

    Args:
        array: 阵列实例
        signal: 信号实例
        azimuth: 信号入射的方位角
        elevation: 信号入射的俯仰角

    Return:
        received_data: M x L 维的矩阵，其中M是阵元数，L是采样点数
    """
    steering_maxtrix = np.zeros(array.num_elements, signal.num_signals)
    steering_maxtrix = array.steering_maxtrix(azimuth, elevation)
    received_data = steering_maxtrix @ signal.gen()
    return received_data

