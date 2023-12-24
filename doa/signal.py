from abc import ABC, abstractmethod
import numpy as np

class Signal(ABC):
    """所有信号的基类

    继承此基类的信号必须要实现gen()方法, 用来产生仿真的采样信号

    Args:
        n (int): 入射信号的数目
        amp (float): 信号的幅度
        nsamples (int): 采样点数
    """
    def __init__(self, n ,amp, nsamples):
        self._n = n
        self._amp = amp
        self._nsamples = nsamples

    @property
    def num_signals(self):
        return self._n

    @abstractmethod
    def gen():
        """产生采样后的信号

        Returns:
            signal (ndarray): 采用后的信号
        """
        raise NotImplementedError()

class ComplexStochasticSignal(Signal):
    def __init__(self, n ,amp, nsamples, fre, fs):
        """随机复信号(随机相位信号的复数形式)

        Args:
            n (int): 入射信号的数目
            amp (float): 信号的幅度
            nsamples (int): 采样点数
            fre (float): 信号的频率
            fs (float): 采样频率
        """
        super().__init__(n, amp, nsamples)

        self._fre = fre
        self._fs = fs

    @property
    def frequency(self):
        """frequency of signal (narrowband)"""
        return self._fre

    def gen(self):
        envelope = self._amp * np.sqrt(1 / 2) *\
            (np.random.randn(self._n, self._nsamples) +\
                1j * np.random.randn(self._n, self._nsamples))  # 复包络
        signal = envelope * np.exp(1j * 2 * np.pi * self._fre / self._fs *\
            np.arange(self._nsamples))
        return signal
