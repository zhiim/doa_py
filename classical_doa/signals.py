from abc import ABC, abstractmethod

import numpy as np


class Signal(ABC):
    """Base class for all signal classes

    Signals that inherit from this base class must implement the gen() method to
    generate simulated sampled signals.
    """

    def __init__(self, nsamples, fs, rng=None):
        self._nsamples = nsamples
        self._fs = fs

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    @property
    def fs(self):
        return self._fs

    @property
    def nsamples(self):
        return self._nsamples

    def set_rng(self, rng):
        """Setting random number generator

        Args:
            rng (np.random.Generator): random generator used to generate random
                numbers
        """
        self._rng = rng

    @abstractmethod
    def gen(self, n, amp=None):
        """Generate sampled signals

        Args:
            n (int): Number of signals
            amp (np.array): Amplitude of the signals (1D array of size n), used
                to define different amplitudes for different signals.
                By default it will generate equal amplitude signal.

        Returns:
            signal (np.array): Sampled signals
        """
        self.n = n

        # Default to generate signals with equal amplitudes
        if amp is None:
            self.amp = np.diag(np.ones(n))
        else:
            self.amp = np.diag(np.squeeze(amp))


class ComplexStochasticSignal(Signal):
    def __init__(self, nsamples, fre, fs, rng=None):
        """Complex stochastic signal (complex exponential form of random phase
        signal)

        Args:
            nsamples (int): Number of sampling points
            fre (float): Signal frequency
            fs (float): Sampling frequency
            rng (np.random.Generator): Random generator used to generate random
                numbers
        """
        super().__init__(nsamples, fs, rng)

        self._fre = fre

    @property
    def frequency(self):
        """Frequency of the signal (narrowband)"""
        return self._fre

    def gen(self, n, amp=None):
        super().gen(n, amp)

        # Generate complex envelope
        envelope = self.amp @ (
            np.sqrt(1 / 2)
            * (
                self._rng.standard_normal(size=(self.n, self._nsamples))
                + 1j * self._rng.standard_normal(size=(self.n, self._nsamples))
            )
        )

        signal = envelope * np.exp(
            -1j * 2 * np.pi * self._fre / self._fs * np.arange(self._nsamples)
        )

        return signal


class ChirpSignal(Signal):
    def __init__(self, nsamples, fs, f0, f1, t1=None, rng=None):
        """Chirp signal

        Args:
            nsamples (int): Number of sampling points
            f0 (np.array): Start frequency at time 0. An 1D array of size n
            f1 (np.array): Frequency at time t1. An 1D array of size n
            t1 (np.array): Time at which f1 is specified. An 1D array of size n
            fs (int | float): Sampling frequency
        """
        super().__init__(nsamples, fs, rng)

        self._f0 = f0
        if t1 is None:
            t1 = np.full(f0.shape, nsamples / fs)
        self._k = (f1 - f0) / t1  # Rate of frequency change

    def gen(self, n, amp=None):
        super().gen(n, amp)

        signal = np.zeros((self.n, self._nsamples), dtype=np.complex128)

        # Generate signal one by one
        for i in range(self.n):
            sampling_time = np.arange(self._nsamples) * 1 / self._fs
            signal[i, :] = np.exp(
                1j
                * 2
                * np.pi
                * (
                    self._f0[i] * sampling_time
                    + 0.5 * self._k[i] * sampling_time**2
                )
            )

        signal = self.amp @ signal

        return signal


class ModBroadSignal(Signal):
    def __init__(self, nsamples, fre_min, fre_max, fs, rng=None):
        """Broadband signal consisting of mulitple narrowband signals modulated
        on different carrier frequencies. Each signal on different carrier
        frequency has a different DOA.

        Args:
            nsamples (int): Number of sampling points
            fs (float): Sampling frequency
            rng (np.random.Generator): Random generator used to generate random
                numbers
        """
        super().__init__(nsamples, fs, rng)

        self._fre_min = fre_min
        self._fre_max = fre_max

    def gen(self, n, amp=None):
        """Generate sampled signals

        Args:
            n (int): Number of signals
            amp (np.array): Amplitude of the signals (1D array of size n), used
                to define different amplitudes for different signals.
                By default it will generate equal amplitude signal.
        """
        super().gen(n, amp)

        # generate random carrier frequencies
        fres = self._rng.uniform(self._fre_min, self._fre_max, size=n)

        # Generate complex envelope
        envelope = self.amp @ (
            np.sqrt(1 / 2)
            * (
                self._rng.standard_normal(size=(self.n, self._nsamples))
                + 1j * self._rng.standard_normal(size=(self.n, self._nsamples))
            )
        )

        signal = envelope * np.exp(
            -1j
            * 2
            * np.pi
            * fres.reshape(-1, 1)
            / self._fs
            * np.arange(self._nsamples)
        )

        return signal
