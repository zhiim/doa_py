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
    def __init__(self, nsamples, fs):
        self._nsamples = nsamples
        self._fs = fs


    @property
    def fs(self):
        return self._fs

    @property
    def nsamples(self):
        return self._nsamples

    @abstractmethod
    def gen(self, n, amp=None):
        """产生采样后的信号

        Args:
            n (int): 信号的数目
            amp (np.array): 信号的幅度(n元素的1d array), 用来定义不同信号的不同
                幅度, 默认为等幅信号

        Returns:
            signal (ndarray): 采用后的信号
        """
        self._n = n

        # 默认生成等幅信号
        if amp is None:
            self._amp = np.diag(np.ones(n))
        else:
            self._amp = np.diag(np.squeeze(amp))

class ComplexStochasticSignal(Signal):
    def __init__(self, nsamples, fre, fs):
        """随机复信号(随机相位信号的复数形式)

        Args:
            n (int): 入射信号的数目
            nsamples (int): 采样点数
            fre (float): 信号的频率
            fs (float): 采样频率
            amp (np.array): 信号的幅度(nx1矩阵), 用来定义不同信号的不同幅度,
                生成同幅度信号时不用考虑
        """
        super().__init__(nsamples, fs)

        self._fre = fre

    @property
    def frequency(self):
        """frequency of signal (narrowband)"""
        return self._fre

    def gen(self, n, amp=None):
        super().gen(n, amp)

        envelope = self._amp @ (np.sqrt(1 / 2) *\
            (np.random.randn(self._n, self._nsamples) +\
                1j * np.random.randn(self._n, self._nsamples)))  # 复包络

        signal = envelope * np.exp(-1j * 2 * np.pi * self._fre / self._fs *\
            np.arange(self._nsamples))

        return signal

class ChirpSignal(Signal):
    def __init__(self, nsamples, fs, f0, f1, t1=None):
        """Chirp signal

        Args:
            n (int): number of signal
            nsamples (int): number of sampling points
            f0 (np.array): start frequency at time 0. An 1d array of size n
            f1 (np.array): frequency at time t1. An 1d array of size n
            t1 (np.array): time at which f1 is specified. An 1d array of size n
            fs (int | float): sampling frequency
            amp (np.array, optional): amplitude of every signal.
        """
        super().__init__(nsamples, fs)

        self._f0 = f0
        if t1 is None:
            t1 = np.full(f0.shape, nsamples / fs)
        self._k = (f1 - f0) / t1  # rate of frequency change

    def gen(self, n, amp=None):
        """Generate n chirp signal

        Args:
            time_start (np.array, optional): time point at which sampling start.
                Defaults to None.
        """
        super().gen(n, amp)

        signal = np.zeros((self._n, self._nsamples), dtype=np.complex_)

        # generate signal one by one
        for i in range(self._n):
            sampling_time = np.arange(self._nsamples) * 1 / self._fs
            signal[i, :] = np.exp(1j * 2 * np.pi * (self._f0[i] * sampling_time\
                                                    + 0.5 * self._k[i]\
                                                        * sampling_time ** 2))

        signal = self._amp @ signal

        return signal
