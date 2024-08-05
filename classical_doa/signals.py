from abc import ABC, abstractmethod

import numpy as np


class Signal(ABC):
    """Base class for all signal classes

    Signals that inherit from this base class must implement the gen() method to
    generate simulated sampled signals.
    """

    def __init__(self, fs, rng=None):
        self._fs = fs

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    @property
    def fs(self):
        return self._fs

    def set_rng(self, rng):
        """Setting random number generator

        Args:
            rng (np.random.Generator): random generator used to generate random
                numbers
        """
        self._rng = rng

    def _get_amp(self, amp, n):
        # Default to generate signals with equal amplitudes
        if amp is None:
            amp = np.diag(np.ones(n))
        else:
            amp = np.diag(np.squeeze(amp))

        return amp

    @abstractmethod
    def gen(self, n, nsamples, amp=None):
        """Generate sampled signals

        Args:
            n (int): Number of signals
            nsamples (int): Number of snapshots
            amp (np.array): Amplitude of the signals (1D array of size n), used
                to define different amplitudes for different signals.
                By default it will generate equal amplitude signal.

        Returns:
            signal (np.array): Sampled signals
        """
        raise NotImplementedError


class ComplexStochasticSignal(Signal):
    def __init__(self, fre, fs, rng=None):
        """Complex stochastic signal (complex exponential form of random phase
        signal)

        Args:
            nsamples (int): Number of sampling points
            fre (float): Signal frequency
            fs (float): Sampling frequency
            rng (np.random.Generator): Random generator used to generate random
                numbers
        """
        super().__init__(fs, rng)

        self._fre = fre

    @property
    def frequency(self):
        """Frequency of the signal (narrowband)"""
        return self._fre

    def gen(self, n, nsamples, amp=None):
        amp = self._get_amp(amp, n)

        # Generate complex envelope
        envelope = amp @ (
            np.sqrt(1 / 2)
            * (
                self._rng.standard_normal(size=(n, nsamples))
                + 1j * self._rng.standard_normal(size=(n, nsamples))
            )
        )

        signal = envelope * np.exp(
            -1j * 2 * np.pi * self._fre / self._fs * np.arange(nsamples)
        )

        return signal


class ChirpSignal(Signal):
    def __init__(self, fs, f0, f1, t1, rng=None):
        """Chirp signal

        Args:
            nsamples (int): Number of sampling points
            f0 (np.array): Start frequency at time 0. An 1D array of size n
            f1 (np.array): Frequency at time t1. An 1D array of size n
            t1 (np.array): Time at which f1 is specified. An 1D array of size n
            fs (int | float): Sampling frequency
        """
        super().__init__(fs, rng)

        self._f0 = f0
        self._k = (f1 - f0) / t1  # Rate of frequency change

    def gen(self, n, nsamples, amp=None):
        amp = self._get_amp(amp, n)

        signal = np.zeros((n, nsamples), dtype=np.complex128)

        # Generate signal one by one
        for i in range(n):
            sampling_time = np.arange(nsamples) * 1 / self._fs
            signal[i, :] = np.exp(
                1j
                * 2
                * np.pi
                * (
                    self._f0[i] * sampling_time
                    + 0.5 * self._k[i] * sampling_time**2
                )
            )

        signal = amp @ signal

        return signal


class MultiCarrierSignal(Signal):
    def __init__(self, fre_min, fre_max, fs, rng=None):
        """Broadband signal consisting of mulitple narrowband signals modulated
        on different carrier frequencies. Each signal on different carrier
        frequency has a different DOA.

        Args:
            fs (float): Sampling frequency
            rng (np.random.Generator): Random generator used to generate random
                numbers
        """
        super().__init__(fs, rng)

        self._fre_min = fre_min
        self._fre_max = fre_max

    def gen(self, n, nsamples, amp=None):
        amp = self._get_amp(amp, n)

        # generate random carrier frequencies
        fres = self._rng.uniform(self._fre_min, self._fre_max, size=n)

        # Generate complex envelope
        envelope = amp @ (
            np.sqrt(1 / 2)
            * (
                self._rng.standard_normal(size=(n, nsamples))
                + 1j * self._rng.standard_normal(size=(n, nsamples))
            )
        )

        signal = envelope * np.exp(
            -1j
            * 2
            * np.pi
            * fres.reshape(-1, 1)
            / self._fs
            * np.arange(nsamples)
        )

        return signal
