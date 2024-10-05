from abc import ABC, abstractmethod

import numpy as np


class Signal(ABC):
    """Base class for all signal classes

    Signals that inherit from this base class must implement the gen() method to
    generate simulated sampled signals.
    """

    def __init__(self, rng=None):
        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

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


class NarrowSignal(Signal):
    def __init__(self, fc, rng=None):
        """Narrowband signal

        Args:
            fc (float): Signal frequency
            rng (np.random.Generator): Random generator used to generate random
                numbers
        """
        self._fc = fc

        super().__init__(rng=rng)

    @property
    def frequency(self):
        """Frequency of narrowband signal"""
        return self._fc

    @abstractmethod
    def gen(self, n, nsamples, amp=None):
        raise NotImplementedError


class ComplexStochasticSignal(NarrowSignal):
    def __init__(self, fc, rng=None):
        """Complex stochastic signal (complex exponential form of random phase
        signal)

        Args:
            fc (float): Signal frequency
            rng (np.random.Generator): Random generator used to generate random
                numbers
        """
        super().__init__(fc, rng)

    def gen(self, n, nsamples, amp=None):
        amp = self._get_amp(amp, n)

        # Generate complex envelope
        signal = amp @ (
            np.sqrt(1 / 2)
            * (
                self._rng.standard_normal(size=(n, nsamples))
                + 1j * self._rng.standard_normal(size=(n, nsamples))
            )
        )
        return signal


class RandomFreqSignal(NarrowSignal):
    def __init__(self, fc, freq_ratio=0.05, rng=None):
        """Random frequency signal

        Args:
            fc (float): Signal frequency
            freq_ratio (float): Ratio of the maximum frequency deviation from fc
            rng (np.random.Generator): Random generator used to generate random
                numbers
        """
        super().__init__(fc, rng)

        assert (
            0 < freq_ratio < 0.1
        ), "This signal must be narrowband: freq_ratio in (0, 0.1)"
        self._freq_ratio = freq_ratio

    def gen(self, n, nsamples, amp=None):
        amp = self._get_amp(amp, n)

        fs = self._fc * self._freq_ratio * 5
        # Generate random frequency signal
        signal = (
            amp
            @ np.exp(
                1j
                * 2
                * np.pi
                * self._rng.uniform(0, self._freq_ratio * self._fc, size=(n, 1))
                / fs
                * np.arange(nsamples)
            )
            * np.exp(1j * self._rng.uniform(0, 2 * np.pi, size=(n, 1)))  # phase
        )
        return signal


class BroadSignal(Signal):
    def __init__(self, f_min, f_max, fs, rng=None):
        self._f_min = f_min
        self._f_max = f_max
        self._fs = fs

        super().__init__(rng=rng)

    @property
    def fs(self):
        return self._fs

    @abstractmethod
    def gen(self, n, nsamples, amp=None):
        raise NotImplementedError

    def _gen_fre_ranges(self, n, min_length_ratio=0.3):
        """Generate frequency ranges for each boardband signal

        Args:
            n (int): Number of signals
            min_length_ratio (float): Minimum length ratio of the frequency
                range in (f_max - f_min)

        Returns:
            ranges (np.array): Frequency ranges for each signal with shape
                (n, 2)
        """
        min_length = (self._f_max - self._f_min) * min_length_ratio
        ranges = np.zeros((n, 2))
        for i in range(n):
            length = self._rng.uniform(min_length, self._f_max - self._f_min)
            start = self._rng.uniform(self._f_min, self._f_max - length)
            ranges[i] = [start, start + length]
        return ranges


class ChirpSignal(BroadSignal):
    def __init__(self, f_min, f_max, fs, rng=None):
        """Chirp signal

        Args:
            f_min (float): Minimum frequency
            f_max (float): Maximum frequency
            fs (int | float): Sampling frequency
            rng (np.random.Generator): Random generator used to generate random
                numbers
        """
        super().__init__(f_min, f_max, fs, rng=rng)

    def gen(self, n, nsamples, amp=None):
        amp = self._get_amp(amp, n)
        fre_ranges = self._gen_fre_ranges(n)

        t = np.arange(nsamples) * 1 / self._fs
        f0 = fre_ranges[:, 0]
        k = (fre_ranges[:, 1] - fre_ranges[:, 0]) / t[-1]

        signal = np.exp(
            1j
            * 2
            * np.pi
            * (f0.reshape(-1, 1) * t + 0.5 * k.reshape(-1, 1) * t**2)
        ) * np.exp(1j * self._rng.uniform(0, 2 * np.pi, size=(n, 1)))  # phase

        signal = amp @ signal
        return signal


class MultiFreqSignal(BroadSignal):
    def __init__(self, f_min, f_max, fs, ncarriers=100, rng=None):
        """Broadband signal consisting of mulitple narrowband signals modulated
        on different carrier frequencies.

        Args:
            f_min (float): Minimum frequency
            f_max (float): Maximum frequency
            fs (int | float): Sampling frequency
            ncarriers (int): Number of carrier frequencies in each broadband
            rng (np.random.Generator): Random generator used to generate random
                numbers
        """
        super().__init__(f_min, f_max, fs, rng=rng)
        self._ncarriers = ncarriers

    def gen(self, n, nsamples, amp=None):
        amp = self._get_amp(amp, n)
        fre_ranges = self._gen_fre_ranges(n)

        # generate random carrier frequencies
        fres = self._rng.uniform(
            fre_ranges[:, 0].reshape(-1, 1),
            fre_ranges[:, 1].reshape(-1, 1),
            size=(n, self._ncarriers),
        )

        signal = np.sum(
            np.exp(
                1j
                * 2
                * np.pi
                * np.repeat(np.expand_dims(fres, axis=2), nsamples, axis=2)
                / self._fs
                * np.arange(nsamples)
            )
            * np.exp(
                1j
                * self._rng.uniform(0, 2 * np.pi, size=(n, self._ncarriers, 1))
            ),
            axis=1,
        )

        signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2))

        signal = amp @ signal

        return signal
