from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

ListLike = Union[npt.NDArray[np.number], list[int | float | complex]]


class Signal(ABC):
    """Base class for all signal classes

    Signals that inherit from this base class must implement the gen() method to
    generate simulated sampled signals.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        # caches used to generate the identical signals
        self._cache = {}

    def set_rng(self, rng: np.random.Generator):
        """Setting random number generator

        Args:
            rng (np.random.Generator): random generator used to generate random
                numbers
        """
        self._rng = rng

    def _set_cache(self, key: str, value: Any):
        """Set cache value

        Args:
            key (str): Cache key
            value (Any): Cache value
        """
        self._cache[key] = value

    def clear_cache(self):
        self._cache = {}

    def _get_amp(
        self,
        amp: Optional[ListLike],
        n: int,
    ) -> npt.NDArray[np.number]:
        # Default to generate signals with equal amplitudes
        if amp is None:
            amp = np.diag(np.ones(n))
        else:
            if not isinstance(amp, np.ndarray):
                amp = np.array(amp)

            amp = np.squeeze(amp)
            if not (amp.ndim == 1 and amp.size == n):
                raise TypeError(
                    "amp should be an 1D array of size n = {}".format(n)
                )

            amp = np.diag(amp)

        return amp

    @abstractmethod
    def gen(
        self,
        n: int,
        nsamples: int,
        amp=Optional[ListLike],
        use_cache: bool = False,
    ) -> npt.NDArray[np.complex128]:
        """Generate sampled signals

        Args:
            n (int): Number of signals
            nsamples (int): Number of snapshots
            amp (np.array): Amplitude of the signals (1D array of size n), used
                to define different amplitudes for different signals.
                By default it will generate equal amplitude signal.
            use_cache (bool): If True, use cache to generate identical signals.
                Default to `False`.

        Returns:
            signal (np.array): Sampled signals
        """
        pass


class NarrowSignal(Signal):
    def __init__(
        self, fc: Union[int, float], rng: Optional[np.random.Generator] = None
    ):
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
    def gen(
        self,
        n: int,
        nsamples: int,
        amp: Optional[ListLike] = None,
        use_cache: bool = False,
    ) -> npt.NDArray[np.complex128]:
        pass


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

    def gen(
        self, n, nsamples, amp=None, use_cache=False
    ) -> npt.NDArray[np.complex128]:
        amp = self._get_amp(amp, n)

        if use_cache and not self._cache == {}:
            # use cache
            real = self._cache["real"]
            imag = self._cache["imag"]
            assert real.shape == (n, nsamples) and imag.shape == (
                n,
                nsamples,
            ), "Cache shape mismatch"
        else:
            # Generate random amp
            real = self._rng.standard_normal(size=(n, nsamples))
            imag = self._rng.standard_normal(size=(n, nsamples))
            self._set_cache("real", real)
            self._set_cache("imag", imag)

        # Generate complex envelope
        signal = amp @ (np.sqrt(1 / 2) * (real + 1j * imag))
        return signal


class RandomFreqSignal(NarrowSignal):
    def __init__(
        self,
        fc: Union[int, float],
        freq_ratio: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ):
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

    def gen(self, n, nsamples, amp=None, use_cache=False):
        amp = self._get_amp(amp, n)

        if use_cache and not self._cache == {}:
            freq = self._cache["freq"]
            phase = self._cache["phase"]
            assert freq.shape == (n, 1) and phase.shape == (
                n,
                1,
            ), "Cache shape mismatch"
        else:
            # Generate random phase signal
            freq = self._rng.uniform(
                0, self._freq_ratio * self._fc, size=(n, 1)
            )
            phase = self._rng.uniform(0, 2 * np.pi, size=(n, 1))
            self._set_cache("freq", freq)
            self._set_cache("phase", phase)

        fs = self._fc * self._freq_ratio * 5
        # Generate random frequency signal
        signal = (
            amp
            @ np.exp(1j * 2 * np.pi * freq / fs * np.arange(nsamples))
            * np.exp(1j * phase)  # phase
        )
        return signal


class BroadSignal(Signal):
    def __init__(
        self,
        f_min: Union[int, float],
        f_max: Union[int, float],
        fs: Union[int, float],
        min_length_ratio: float = 0.1,
        no_overlap: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self._f_min = f_min
        self._f_max = f_max
        self._fs = fs
        self._min_length_ratio = min_length_ratio
        self._no_overlap = no_overlap

        super().__init__(rng=rng)

    @property
    def fs(self):
        return self._fs

    @property
    @abstractmethod
    def f_min(self):
        pass

    @property
    @abstractmethod
    def f_max(self):
        pass

    @abstractmethod
    def gen(
        self,
        n: int,
        nsamples: int,
        amp: Optional[ListLike] = None,
        use_cache: bool = False,
        delay: Optional[Union[npt.NDArray, int, float]] = None,
    ) -> npt.NDArray[np.complex128]:
        """Generate sampled signals

        Args:
            n (int): Number of signals
            nsamples (int): Number of snapshots
            amp (np.array): Amplitude of the signals (1D array of size n), used
                to define different amplitudes for different signals.
                By default it will generate equal amplitude signal.
            use_cache (bool): If True, use cache to generate identical signals.
                Default to `False`.
            delay (float | None): If not None, apply delay to all signals.

        Returns:
            signal (np.array): Sampled signals
        """
        pass

    def _gen_fre_bands(self, n: int):
        """Generate frequency ranges for each boardband signal

        Args:
            n (int): Number of signals

        Returns:
            ranges (np.array): Frequency ranges for each signal with shape
                (n, 2)
        """
        if self._no_overlap:
            return self._gen_fre_bands_no_overlapping(n)
        return self._gen_fre_bands_overlapping(n)

    def _gen_fre_bands_overlapping(self, n: int):
        """Generate frequency bands may overlapping."""
        min_length = (self._f_max - self._f_min) * self._min_length_ratio
        bands = np.zeros((n, 2))
        for i in range(n):
            length = self._rng.uniform(min_length, self._f_max - self._f_min)
            start = self._rng.uniform(self._f_min, self._f_max - length)
            bands[i] = [start, start + length]
        return bands

    def _gen_fre_bands_no_overlapping(self, n: int):
        """Generate non-overlapping frequency bands."""
        max_length = (self._f_max - self._f_min) // n
        min_length = max_length * self._min_length_ratio

        bands = np.zeros((n, 2))

        for i in range(n):
            length = self._rng.uniform(min_length, max_length)
            start = self._rng.uniform(
                self._f_min + i * max_length,
                self._f_min + (i + 1) * max_length - length,
            )
            new_band = [start, start + length]

            bands[i] = new_band

        return bands


class ChirpSignal(BroadSignal):
    def __init__(
        self,
        f_min,
        f_max,
        fs,
        min_length_ratio: float = 0.1,
        no_overlap: bool = False,
        rng=None,
    ):
        """Chirp signal

        Args:
            f_min (float): Minimum frequency
            f_max (float): Maximum frequency
            fs (int | float): Sampling frequency
            min_length_ratio (float): Minimum length ratio of the frequency
                band in (f_max - f_min)
            no_overlap (bool): If True, generate signals with non-overlapping
                bands
            rng (np.random.Generator): Random generator used to generate random
                numbers
        """
        super().__init__(
            f_min, f_max, fs, min_length_ratio, no_overlap, rng=rng
        )

    @property
    def f_min(self):
        if "fre_ranges" not in self._cache:
            raise ValueError("fre_ranges not in cache")
        return np.min(self._cache["fre_ranges"][:, 0])

    @property
    def f_max(self):
        if "fre_ranges" not in self._cache:
            raise ValueError("fre_ranges not in cache")
        return np.max(self._cache["fre_ranges"][:, 1])

    def gen(
        self,
        n,
        nsamples,
        amp=None,
        use_cache=False,
        delay=None,
    ) -> npt.NDArray[np.complex128]:
        amp = self._get_amp(amp, n)

        # use cache
        if use_cache and not self._cache == {}:
            fre_ranges = self._cache["fre_ranges"]
            phase = self._cache["phase"]
            assert fre_ranges.shape == (n, 2) and phase.shape == (
                n,
                1,
            ), "Cache shape mismatch"
        # generate new and write to cache
        else:
            fre_ranges = self._gen_fre_bands(n)
            phase = self._rng.uniform(0, 2 * np.pi, size=(n, 1))
            self._set_cache("fre_ranges", fre_ranges)
            self._set_cache("phase", phase)

        t = np.arange(nsamples) * 1 / self._fs

        # start freq
        f0 = fre_ranges[:, 0]
        # freq move to f1 in t
        k = (fre_ranges[:, 1] - fre_ranges[:, 0]) / t[-1]

        if delay is not None:
            if isinstance(delay, (int, float)):
                delay = np.ones(n) * delay
            t = t + delay.reshape(-1, 1)

        signal = np.exp(
            1j
            * 2
            * np.pi
            * (f0.reshape(-1, 1) * t + 0.5 * k.reshape(-1, 1) * t**2)
        ) * np.exp(1j * phase)

        signal = amp @ signal

        return signal


class MultiFreqSignal(BroadSignal):
    def __init__(
        self,
        f_min: Union[int, float],
        f_max: Union[int, float],
        fs: Union[int, float],
        min_length_ratio: float = 0.1,
        no_overlap: bool = False,
        rng: Optional[np.random.Generator] = None,
        ncarriers: int = 100,
    ):
        """Broadband signal consisting of mulitple narrowband signals modulated
        on different carrier frequencies.

        Args:
            f_min (float): Minimum frequency
            f_max (float): Maximum frequency
            fs (int | float): Sampling frequency
            min_length_ratio (float): Minimum length ratio of the frequency
                band in (f_max - f_min)
            no_overlap (bool): If True, generate signals with non-overlapping
                bands
            rng (np.random.Generator): Random generator used to generate random
                numbers
            ncarriers (int): Number of carrier frequencies in each broadband
        """
        super().__init__(
            f_min, f_max, fs, min_length_ratio, no_overlap, rng=rng
        )

        self._ncarriers = ncarriers

    @property
    def f_min(self):
        if "fres" not in self._cache:
            raise ValueError("fres not in cache")
        return np.min(self._cache["fres"])

    @property
    def f_max(self):
        if "fres" not in self._cache:
            raise ValueError("fres not in cache")
        return np.max(self._cache["fres"])

    def gen(
        self,
        n,
        nsamples,
        amp=None,
        use_cache=False,
        delay=None,
    ) -> npt.NDArray[np.complex128]:
        amp = self._get_amp(amp, n)
        """Generate sampled signals

        Args:
            n (int): Number of signals
            nsamples (int): Number of snapshots
            amp (np.array): Amplitude of the signals (1D array of size n), used
                to define different amplitudes for different signals.
                By default it will generate equal amplitude signal.
            use_cache (bool): If True, use cache to generate identical signals.
                Default to `False`.
            delay (float | None): If not None, apply delay to all signals.

        Returns:
            signal (np.array): Sampled signals
        """

        if use_cache and not self._cache == {}:
            fres = self._cache["fres"]
            phase = self._cache["phase"]
            assert fres.shape == (n, self._ncarriers) and phase.shape == (
                n,
                self._ncarriers,
                1,
            ), "Cache shape mismatch"
        else:
            fre_ranges = self._gen_fre_bands(n)
            # generate random carrier frequencies
            fres = self._rng.uniform(
                fre_ranges[:, 0].reshape(-1, 1),
                fre_ranges[:, 1].reshape(-1, 1),
                size=(n, self._ncarriers),
            )
            phase = self._rng.uniform(
                0, 2 * np.pi, size=(n, self._ncarriers, 1)
            )
            self._set_cache("fres", fres)
            self._set_cache("phase", phase)

        t = np.arange(nsamples) * (1 / self._fs)

        if delay is not None:
            if isinstance(delay, (int, float)):
                delay = np.ones(n) * delay
            t = t + delay.reshape(-1, 1)  # t is broadcasted to (n, nsamples)
            # let t able to be broadcasted where calculating `signal`
            t = np.expand_dims(t, axis=1)

        signal = np.sum(
            np.exp(
                1j
                * 2
                * np.pi
                * np.repeat(np.expand_dims(fres, axis=2), nsamples, axis=2)
                * t
            )
            * np.exp(1j * phase),
            axis=1,
        )

        # norm signal power to 1
        signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2))

        signal = amp @ signal

        return signal


class MixedSignal(BroadSignal):
    def __init__(
        self,
        f_min: Union[int, float],
        f_max: Union[int, float],
        fs: Union[int, float],
        min_length_ratio: float = 0.1,
        no_overlap: bool = False,
        rng: Optional[np.random.Generator] = None,
        base: Literal["chirp", "multifreq"] = "chirp",
        ncarriers: int = 100,
    ):
        """Narrorband and broadband mixed signal

        Args:
            f_min (float): Minimum frequency
            f_max (float): Maximum frequency
            fs (int | float): Sampling frequency
            min_length_ratio (float): Minimum length ratio of the frequency
                band in (f_max - f_min)
            no_overlap (bool): If True, generate signals with non-overlapping
                bands
            rng (np.random.Generator): Random generator used to generate random
                numbers
            base (str): Type of base signal, either 'chirp' or 'multifreq'
            ncarriers (int): Only for `multifreq` base. Number of carrier
                frequencies in each broadband

        Raises:
            ValueError: If base is not 'chirp' or 'multifreq'
        """
        if base not in ["chirp", "multifreq"]:
            raise ValueError("base must be either 'chirp' or 'multifreq'")
        if base == "chirp":
            self._base = ChirpSignal(
                f_min, f_max, fs, min_length_ratio, no_overlap, rng
            )
        elif base == "multifreq":
            self._base = MultiFreqSignal(
                f_min, f_max, fs, min_length_ratio, no_overlap, rng, ncarriers
            )

        super().__init__(
            f_min, f_max, fs, min_length_ratio, no_overlap, rng=rng
        )

    def clear_cache(self):
        super().clear_cache()
        self._base.clear_cache()

    @property
    def f_min(self):
        return np.min([np.min(self._cache["narrow_freqs"]), self._base.f_min])

    @property
    def f_max(self):
        return np.max([np.max(self._cache["narrow_freqs"]), self._base.f_max])

    def gen(
        self,
        n: int,
        nsamples: int,
        amp: Optional[ListLike] = None,
        use_cache=False,
        delay: Optional[Union[npt.NDArray, int, float]] = None,
        m: Optional[int] = None,
        narrow_idx: Union[npt.NDArray[np.int_], list[int], None] = None,
    ):
        """Generate sampled signals

        Args:
            n (int): Number of all signals (narrowband and broadband)
            nsamples (int): Number of snapshots
            amp (np.array): Amplitude of the signals (1D array of size n), used
                to define different amplitudes for different signals.
                By default it will generate equal amplitude signal.
            use_cache (bool): If True, use cache to generate identical signals.
                Default to `False`.
            delay (float | None): If not None, apply delay to all signals.
            m (int): Number of narrowband signals inside `n`. If set to `None`,
                it will use a random int smaller than n
            narrow_idx (array): index of where narrowband signal is located in n
                signals
        """
        if m is None:
            m = self._rng.integers(1, n)
        else:
            if m >= n:
                raise ValueError(
                    "Number of narrowband signals must be less than n"
                )

        amp = self._get_amp(amp, n)

        if use_cache and not self._cache == {}:
            narrow_freqs = self._cache["narrow_freqs"]
            phase = self._cache["phase"]
            narrow_idx = self._cache["narrow_idx"]
            assert narrow_freqs.shape == (m, 1), "Cache shape mismatch"
            assert phase.shape == (m, 1), "Cache shape mismatch"
            assert isinstance(narrow_idx, np.ndarray)
        else:
            narrow_freqs = self._rng.uniform(
                self._f_min, self._f_max, size=m
            ).reshape(-1, 1)
            phase = self._rng.uniform(0, 2 * np.pi, size=(m, 1))
            if narrow_idx is None:
                narrow_idx = self._rng.choice(n, m, replace=False)
            narrow_idx = np.array(narrow_idx)
            assert len(narrow_idx) == m, "narrow_idx length mismatch"
            self._set_cache("narrow_freqs", narrow_freqs)
            self._set_cache("phase", phase)
            self._set_cache("narrow_idx", narrow_idx)

        if delay is not None:
            if isinstance(delay, (int, float)):
                delay = np.ones(n) * delay
        else:
            delay = np.zeros(n)

        broad_idx = ~np.isin(np.arange(n), narrow_idx)

        # generate narrowband signals
        t = np.arange(nsamples) * (1 / self._fs)
        t = t + delay.reshape(-1, 1)[narrow_idx]

        narrow_s = (
            np.exp(1j * 2 * np.pi * narrow_freqs * t)  # sine wave
            * np.exp(1j * phase)  # phase
        )

        # generate broadband signals
        broad_s = self._base.gen(
            n=n - m,
            nsamples=nsamples,
            use_cache=use_cache,
            delay=delay.reshape(-1, 1)[broad_idx],
        )

        # combine narrowband and broadband signals
        signal = np.zeros((n, nsamples), dtype=np.complex128)
        signal[narrow_idx] = narrow_s
        signal[broad_idx] = broad_s

        signal = amp @ signal

        return signal
