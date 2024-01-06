from abc import ABC, abstractmethod
import numpy as np

C = 3e8  # 波速

class Array(ABC):
    def __init__(self, element_positon):
        self._element_positon = element_positon

    def steering_vector(self, fre, azimuth, elevation=None, unit="deg"):
        """计算某一入射角度对应的导向矢量

        Args:
            fre (float): 入射信号的频率
            azimuth (float): 入射信号的方位角
            elevation (float): 入射信号的俯仰角. Defaults to None, 如果
                `elevation` 为 None 只考虑一维信号.
            unit (str): {'rad', 'deg'}, 入射角的角度制，'rad'代表弧度，
                'deg'代表角度，默认为'deg'

        Returns:
            steering_vector (ndarray): 导向矢量
        """
        raise NotImplementedError()

    def received_signal(self, signal, snr, angle_incidence,
                        broadband=False, unit="deg"):
        """Generate array received signal based on array signal model

        如果`broadband`为True, 生成宽带信号的仿真.

        Args:
            signal : Signal 类实例化的对象
            snr: 信噪比
            angle_incidence : 入射角度. 如果只考虑方位角, `angle_incidence`是
                一个1xN维矩阵; 如果考虑二维, `angle_incidence`是一个2xN维矩阵,
                其中第一行为方位角, 第二行为俯仰角.
            unit : 角度的单位制, `rad`代表弧度制, `deg`代表角度制. Defaults to
                'deg'.
        """
        # 将角度转换为弧度
        if unit == 'deg':
            angle_incidence = angle_incidence / 180 * np.pi

        if broadband is False:
            received = self._gen_narrowband(signal, snr, angle_incidence)
        else:
            received = self._gen_broadband(signal, snr, angle_incidence)

        return received

    @abstractmethod
    def _gen_narrowband(self, signal, snr, angle_incidence):
        """Generate narrowband received signal

        `azimuth` and `elevation` are already in radians
        """
        raise NotImplementedError()

    @abstractmethod
    def _gen_broadband(self, signal, snr, angle_incidence):
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
        # 阵元位置应该是一个Mx1维矩阵，用于后续计算导向矢量
        super().__init__(np.arange(m).reshape(-1, 1) * dd)

    def _gen_narrowband(self, signal, snr, angle_incidence):
        """ULA时, angle_incidence应该是一个对应方位角的行向量"""
        azimuth = angle_incidence.reshape(1, -1)

        # 计算时延矩阵
        matrix_tau = 1 / C * self._element_positon @ np.sin(azimuth)
        # 计算流形矩阵
        manifold_matrix = np.exp(-1j * 2 * np.pi * signal.frequency *
                                 matrix_tau)

        incidence_signal = signal.gen()

        received = manifold_matrix @ incidence_signal

        noise = 1 / np.sqrt(10 ** (snr / 10)) * np.mean(np.abs(received)) *\
            1 / np.sqrt(2) * (np.random.randn(*received.shape) +\
                              1j * np.random.randn(*received.shape))

        received = received + noise

        return received

    def _gen_broadband(self, signal, snr, angle_incidence):
        num_antennas = self._element_positon.size

        incidence_signal = signal.gen()
        num_snapshots = incidence_signal.shape[1]

        # generate array signal in frequency domain
        signal_fre_domain = np.fft.fft(incidence_signal, axis=1)

        azimuth = angle_incidence.reshape(1, -1)
        matrix_tau = 1 / C * self._element_positon @ np.sin(azimuth)

        received_fre_domain = np.zeros((num_antennas, num_snapshots),
                                       dtype=np.complex_)
        for i in range(num_snapshots):
            if i > num_snapshots // 2:
                # 负频率部分
                fre_point = i * signal.fs / num_snapshots - signal.fs
            else:
                # 正频率部分
                fre_point = i * signal.fs / num_snapshots

            manifold_fre = np.exp(-1j * 2 * np.pi * fre_point * matrix_tau)

            # calculate array received signal at every frequency point
            received_fre_domain[:, i] = manifold_fre @ signal_fre_domain[:, i]

        received = np.fft.ifft(received_fre_domain, axis=1)

        noise = 1 / np.sqrt(10 ** (snr / 10)) * np.mean(np.abs(received)) *\
            1 / np.sqrt(2) * (np.random.randn(*received.shape) +\
                              1j * np.random.randn(*received.shape))
        received = received + noise

        return received

