import numpy as np

class ComplexStochasticSignal:
    def __init__(self, amp: float, fre: float, fs: float, nsamples: int):
        """随机复信号(随机相位信号的复数形式)

        Args:
            amp: 信号的幅度
            fre: 信号的频率
            fs: 采样频率
            nsamples: 采样点数
        """
        self._amp = amp
        self._fre = fre
        self._nsamples = nsamples
        self._fs = fs

    def gen(self):
        envelope = self._amp * np.sqrt(1 / 2) *\
            (np.random.randn(1, self._nsamples) +\
                1j * np.random.randn(1, self._nsamples))  # 复包络
        return envelope * np.exp(1j * 2 * np.pi * self._fre / self._fs *\
            np.arange(self._nsamples))

