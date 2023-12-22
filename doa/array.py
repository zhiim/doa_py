from abc import ABC, abstractmethod
import numpy as np

C = 3e8  # 波速

class Array(ABC):
    def __init__(self, element_positon):
        self._element_positon = element_positon

    @property
    @abstractmethod
    def num_elements(self):
        """阵元数"""
        raise NotImplementedError()

    @abstractmethod
    def steering_vector(self, fre, azimuth, elevation=None, unit='rad'):
        """计算某一入射角度对应的导向矢量

        Args:
            fre (float): 入射信号的频率
            azimuth (float): 入射信号的方位角
            elevation (float): 入射信号的俯仰角. Defaults to None, 如果
                `elevation` 为 None 只考虑一维信号.
            unit (str): {'rad', 'deg'}, 入射角的角度制，'rad'代表弧度，
                'deg'代表角度，默认为'rad'

        Returns:
            steering_vector (ndarray): 导向矢量
        """
        raise NotImplementedError()

    def received_signal(self, signal, azimuth, elevation=None, broadband=False,
                        unit='rad'):
        """Generate array received signal based on array signal model

        如果`broadband`为True, 生成宽带信号的仿真.

        Args:
            signal : Signal 类实例化的对象
            azimuth : 方位角向量, 对应不同入射信号的方位角
            elevation : 俯仰角向量, 对应不同入射信号的俯仰角. Defaults to None,
                如果 `elevation` 为 None 只考虑一维信号.
            unit : 角度的单位制, `rad`代表弧度制, `deg`代表角度制. Defaults to
                'rad'.
        """
        azimuth = self._deg_to_rad(azimuth, unit)
        elevation = self._deg_to_rad(elevation, unit)

        if broadband is False:
            self._gen_narrowband(signal, azimuth, elevation)
        else:
            self._gen_broadband(signal, azimuth, elevation)


    def _deg_to_rad(self, angle, unit):
        """将角度转换为弧度"""
        if unit == 'deg' and angle is not None:
            angle = angle / 180 * np.pi
        return angle

    @abstractmethod
    def _gen_narrowband(self, signal, azimuth, elevation=None):
        """Generate narrowband received signal

        `azimuth` and `elevation` are already in radians
        """
        raise NotImplementedError()

    @abstractmethod
    def _gen_broadband(self, signal, azimuth, elevation=None):
        """Generate broadband received signal

        `azimuth` and `elevation` are already in radians
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

    def steering_vector(self, fre, azimuth, elevation=None, unit='rad'):
        azimuth = self._deg_to_rad(azimuth, unit)

        tau = 1 / C * self._element_positon * np.sin(azimuth)  # 时延
        steering_vector = np.exp(-1j * 2 * np.pi * fre * tau)

        return steering_vector

    def _gen_narrowband(self, signal, azimuth, elevation=None):
        matrix_tau = 1 / C * self._element_positon * np.sin(azimuth)  # 时延
        # 计算流形矩阵
        manifold_matrix = np.exp(-1j * 2 * np.pi * signal.frequency *
                                 matrix_tau)
        received = manifold_matrix @ signal.gen()

        return received

    def _gen_broadband(self, signal, azimuth, elevation=None):
        # 暂时不定义
        pass

