from abc import ABC, abstractmethod
import numpy as np

class Signal(ABC):
    """所有信号的基类

    继承此基类的信号必须要实现gen()方法, 用来产生仿真的采样信号

    Args:
        n (int): 入射信号的数目
        amp (np.array): 信号的幅度(n元素的1d array), 用来定义不同信号的不同幅度,
            生成同幅度信号时不用考虑
        nsamples (int): 采样点数
    """
    def __init__(self, n, nsamples, amp=None):
        self._n = n
        self._nsamples = nsamples

        # 默认生成等幅信号
        if amp is None:
            self._amp = np.diag(np.ones(n))
        else:
            self._amp = np.diag(np.squeeze(amp))

    @abstractmethod
    def gen():
        """产生采样后的信号

        Returns:
            signal (ndarray): 采用后的信号
        """
        raise NotImplementedError()

class ComplexStochasticSignal(Signal):
    def __init__(self, n , nsamples, fre, fs, amp=None):
        """随机复信号(随机相位信号的复数形式)

        Args:
            n (int): 入射信号的数目
            nsamples (int): 采样点数
            fre (float): 信号的频率
            fs (float): 采样频率
            amp (np.array): 信号的幅度(nx1矩阵), 用来定义不同信号的不同幅度,
                生成同幅度信号时不用考虑
        """
        super().__init__(n, nsamples, amp)

        self._fre = fre
        self._fs = fs

    @property
    def frequency(self):
        """frequency of signal (narrowband)"""
        return self._fre

    def gen(self):
        envelope = self._amp @ (np.sqrt(1 / 2) *\
            (np.random.randn(self._n, self._nsamples) +\
                1j * np.random.randn(self._n, self._nsamples)))  # 复包络

        signal = envelope * np.exp(-1j * 2 * np.pi * self._fre / self._fs *\
            np.arange(self._nsamples))

        return signal
