from abc import ABC, abstractmethod
import numpy as np

C = 3e8  # 光速

class Array(ABC):
    def __init__(self, element_positon):
        self._element_positon = element_positon

    @property
    @abstractmethod
    def num_elements(self):
        """阵元数"""
        raise NotImplementedError()

    @abstractmethod
    def steering_vector(self, fre: float, azimuth: float,
                        elevation: float, unit: str = 'rad'):
        """计算某一入射角度对应的流型矩阵

        Args:
            fre (float): 入射信号的频率
            azimuth (float): 入射信号的方位角
            elevation (float): 入射信号的俯仰角
            unit (str): {'rad', 'deg'}, 入射角的角度制，'rad'代表弧度，
                'deg'代表角度，默认为'rad'

        Returns:
            steering_vector (ndarray): 流型矩阵
        """
        raise NotImplementedError()


class UniformLinearArray(Array):
    def __init__(self, m: int, dd: float):
        """均匀线阵

        Args:
            m (int): number of antenna elements
            dd (float): distance between adjacent antennas
        """
        super().__init__(np.arange(m) * dd)

    @property
    def num_elements(self):
        return self._element_positon.shape[0]

    def steering_vector(self, fre: float, azimuth: float,
                        elevation: float, unit: str = 'rad'):
        if unit == 'deg':
            azimuth = azimuth / 180 * np.pi  # 转换为弧度

        tau = 1 / C * self._element_positon * np.sin(azimuth)  # 时延
        steering_vector = np.exp(-1j * 2 * np.pi * fre * tau)

        return steering_vector

