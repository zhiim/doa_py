from abc import ABC, abstractmethod

import numpy as np


class Signal(ABC):
    """所有信号的基类

    继承此基类的信号必须要实现gen()方法, 用来产生仿真的采样信号
    """
    def __init__(self, nsamples, fs, rng=None):
        self._nsamples = nsamples
        self._fs = fs

        if rng is None:
            self._rng = np.random.default_rng()

    @property
    def fs(self):
        return self._fs

    @property
    def nsamples(self):
        return self._nsamples

    def set_rng(self, rng):
        """Setting random number generator

        Args:
            rng (np.random.Generator): random generator used to generator random
        """
        self._rng = rng

    @abstractmethod
    def gen(self, n, amp=None):
        """产生采样后的信号

        Args:
            n (int): 信号的数目
            amp (np.array): 信号的幅度(n元素的1d array), 用来定义不同信号的不同
                幅度, 默认为等幅信号

        Returns:
            signal (np.array): 采用后的信号
        """
        self.n = n

        # 默认生成等幅信号
        if amp is None:
            self.amp = np.diag(np.ones(n))
        else:
            self.amp = np.diag(np.squeeze(amp))


class ComplexStochasticSignal(Signal):
    def __init__(self, nsamples, fre, fs, rng=None):
        """随机复信号(随机相位信号的复数形式)

        Args:
            nsamples (int): 采样点数
            fre (float): 信号的频率
            fs (float): 采样频率
            rng (np.random.Generator): random generator used to generator random
        """
        super().__init__(nsamples, fs, rng)

        self._fre = fre

    @property
    def frequency(self):
        """frequency of signal (narrowband)"""
        return self._fre

    def gen(self, n, amp=None):
        super().gen(n, amp)

        # 产生复包络
        envelope = self.amp @ (np.sqrt(1 / 2) *\
                (self._rng.standard_normal(size=(self.n, self._nsamples)) +\
                 1j * self._rng.standard_normal(size=(self.n, self._nsamples))))

        signal = envelope * np.exp(-1j * 2 * np.pi * self._fre / self._fs *
                                   np.arange(self._nsamples))

        return signal


class ChirpSignal(Signal):
    def __init__(self, nsamples, fs, f0, f1, t1=None, rng=None):
        """Chirp signal

        Args:
            nsamples (int): number of sampling points
            f0 (np.array): start frequency at time 0. An 1d array of size n
            f1 (np.array): frequency at time t1. An 1d array of size n
            t1 (np.array): time at which f1 is specified. An 1d array of size n
            fs (int | float): sampling frequency
        """
        super().__init__(nsamples, fs, rng)

        self._f0 = f0
        if t1 is None:
            t1 = np.full(f0.shape, nsamples / fs)
        self._k = (f1 - f0) / t1  # rate of frequency change

    def gen(self, n, amp=None):
        super().gen(n, amp)

        signal = np.zeros((self.n, self._nsamples), dtype=np.complex_)

        # generate signal one by one
        for i in range(self.n):
            sampling_time = np.arange(self._nsamples) * 1 / self._fs
            signal[i, :] = np.exp(1j * 2 * np.pi * (self._f0[i] * sampling_time
                                                    + 0.5 * self._k[i]
                                                    * sampling_time ** 2))

        signal = self.amp @ signal

        return signal
