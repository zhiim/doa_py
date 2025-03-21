import warnings
from abc import ABC
from typing import Literal

import numpy as np
from scipy.linalg import toeplitz
from typing_extensions import override

from .signals import BroadSignal, NarrowSignal, RandomFreqSignal, Signal

C = 3e8  # wave speed


class Array(ABC):
    def __init__(
        self,
        element_position_x,
        element_position_y,
        element_position_z,
        rng=None,
    ):
        """element position should be defined in 3D (x, y, z) coordinate
        system"""
        self._element_position = np.vstack(
            (element_position_x, element_position_y, element_position_z)
        ).T

        self._ideal_position = self._element_position.copy()

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    @property
    def num_antennas(self):
        return self._element_position.shape[0]

    @property
    def array_position(self):
        return self._ideal_position

    def set_rng(self, rng):
        """Setting random number generator

        Args:
            rng (np.random.Generator): random generator used to generator random
        """
        self._rng = rng

    def _unify_unit(self, variable, unit):
        if unit == "deg":
            variable = variable / 180 * np.pi

        return variable

    def steering_vector(self, fre, angle_incidence, unit="deg"):
        """Calculate steering vector corresponding to the angle of incidence

        Args:
            fre (float): Frequency of carrier wave
            angle_incidence (float | np.ndarray): Incidence angle. If only
                azimuth is considered, `angle_incidence` is a 1xN dimensional
                matrix; if two dimensions are considered, `angle_incidence` is
                a 2xN dimensional matrix, where the first row is the azimuth and
                the second row is the elevation angle.
            unit: The unit of the angle, `rad` represents radian,
                `deg` represents degree. Defaults to 'deg'.

        Returns:
            If `angle_incidence` corresponds to single signal, return a steering
            vector of dimension `Mx1`. If `angle_incidence` is doa of k signals,
            return a steering maxtrix of dimension `Mxk`
        """
        if np.squeeze(angle_incidence).ndim == 1 or angle_incidence.size == 1:
            angle_incidence = np.vstack(
                (angle_incidence.reshape(1, -1), np.zeros(angle_incidence.size))
            )

        angle_incidence = self._unify_unit(
            np.reshape(angle_incidence, (2, -1)), unit
        )

        cos_cos = np.cos(angle_incidence[0]) * np.cos(angle_incidence[1])
        sin_cos = np.sin(angle_incidence[0]) * np.cos(angle_incidence[1])
        sin_ = np.sin(angle_incidence[1])

        # only the ideal position is known by algorithms
        time_delay = (
            1 / C * self._ideal_position @ np.vstack((cos_cos, sin_cos, sin_))
        )
        steering_vector = np.exp(-1j * 2 * np.pi * fre * time_delay)

        return steering_vector

    def _steering_vector_with_error(self, fre, angle_incidence, unit="deg"):
        if np.squeeze(angle_incidence).ndim == 1 or angle_incidence.size == 1:
            angle_incidence = np.vstack(
                (angle_incidence.reshape(1, -1), np.zeros(angle_incidence.size))
            )

        angle_incidence = self._unify_unit(
            np.reshape(angle_incidence, (2, -1)), unit
        )

        cos_cos = np.cos(angle_incidence[0]) * np.cos(angle_incidence[1])
        sin_cos = np.sin(angle_incidence[0]) * np.cos(angle_incidence[1])
        sin_ = np.sin(angle_incidence[1])

        # used to generate received signal using the actual position
        time_delay = (
            1 / C * self._element_position @ np.vstack((cos_cos, sin_cos, sin_))
        )
        steering_vector = np.exp(-1j * 2 * np.pi * fre * time_delay)

        # applying mutual coupling effects
        if (
            hasattr(self, "_coupling_matrix")
            and self._coupling_matrix is not None
        ):
            steering_vector = self._coupling_matrix @ steering_vector

        return steering_vector

    def add_position_error(self):
        """Add position error to the array

        Args:
            error_std (float): Standard deviation of the position error
            error_type (str): Type of the error, `gaussian` or `uniform`
        """
        warnings.warn(
            "This method is not implemented for {}".format(type(self).__name__),
            UserWarning,
        )

    def add_mutual_coupling(self, coupling_matrix=None):
        """Add mutual coupling effects to the array

        Args:
            coupling_matrix (np.ndarray): A square matrix representing mutual
                coupling between array elements. Should be of size
                (num_antennas, num_antennas).
        """
        if coupling_matrix is None:
            coupling_matrix = self.get_default_coupling_matrix()

        if coupling_matrix.shape != (self.num_antennas, self.num_antennas):
            raise ValueError(
                f"Coupling matrix shape {coupling_matrix.shape} does not match "
                f"the number of antennas {self.num_antennas}"
            )

        self._coupling_matrix = coupling_matrix

    def get_default_coupling_matrix(self):
        warnings.warn(
            "Default mutual coupling matrix is not provided for {}, no mutual "
            "coupling will be considered".format(type(self).__name__),
            UserWarning,
        )
        return np.eye(self.num_antennas)

    def add_correlation_matrix(self, correlation_matrix=None):
        """Add spatial correlation matrix to the array, which is used to
        generate spatially correlated noise.

        If this method is not called, use the spatially and temporally
        uncorrelated noise.

        If the `correlation_matrix` is not provided, use the default correlation
        matrix.

        Args:
            correlation_matrix (np.ndarray): A square matrix representing
                spatial correlation between array elements. Should be of size
                (num_antennas, num_antennas). Defaults to None.
        """
        if correlation_matrix is None:
            correlation_matrix = self.get_default_correlation_matrix()

        if correlation_matrix.shape != (self.num_antennas, self.num_antennas):
            raise ValueError(
                f"Correlation matrix shape {correlation_matrix.shape} does not "
                f"match the number of antennas {self.num_antennas}"
            )

        self._correlation_matrix = correlation_matrix

    def get_default_correlation_matrix(self):
        """Use identity matrix as the default correlation matrix, which results
        in spatially white noise.

        You can override this method to provide a different default correlation.
        """
        warnings.warn(
            "Default correlation matrix is not provided for {}, the generated "
            "noise will be spatially white noise".format(type(self).__name__),
            UserWarning,
        )
        return np.eye(self.num_antennas)

    def received_signal(
        self,
        signal: Signal,
        angle_incidence,
        snr=None,
        nsamples=100,
        amp=None,
        unit="deg",
        use_cache=False,
        calc_method: Literal["delay", "fft"] = "delay",
        **kwargs,
    ):
        """Generate array received signal based on array signal model

        If `broadband` is set to True, generate array received signal based on
        broadband signal's model.

        Args:
            signal: An instance of the `Signal` class
            angle_incidence: Incidence angle. If only azimuth is considered,
                `angle_incidence` is a 1xN dimensional matrix; if two dimensions
                are considered, `angle_incidence` is a 2xN dimensional matrix,
                where the first row is the azimuth and the second row is the
                elevation angle.
            snr: Signal-to-noise ratio. If set to None, no noise will be added
            nsamples (int): Number of snapshots, defaults to 100
            amp: The amplitude of each signal, 1d numpy array
            unit: The unit of the angle, `rad` represents radian,
                `deg` represents degree. Defaults to 'deg'.
            use_cache (bool): If True, use cache to generate identical signals
                (noise is random). Default to `False`.
            calc_method (str): Only used when generate broadband signal.
                Generate broadband signal in frequency domain, or time domain
                using delay. Defaults to `delay`.
        """
        # Convert the angle from degree to radians
        angle_incidence = self._unify_unit(angle_incidence, unit)

        if isinstance(signal, BroadSignal):
            received = self._gen_broadband(
                signal=signal,
                snr=snr,
                nsamples=nsamples,
                angle_incidence=angle_incidence,
                amp=amp,
                use_cache=use_cache,
                calc_method=calc_method,
                **kwargs,
            )
        if isinstance(signal, NarrowSignal):
            received = self._gen_narrowband(
                signal=signal,
                snr=snr,
                nsamples=nsamples,
                angle_incidence=angle_incidence,
                amp=amp,
                use_cache=use_cache,
            )

        return received

    def _gen_narrowband(
        self,
        signal: NarrowSignal,
        snr,
        nsamples,
        angle_incidence,
        amp,
        use_cache=False,
    ):
        """Generate narrowband received signal

        `azimuth` and `elevation` are already in radians
        """
        if angle_incidence.ndim == 1:
            num_signal = angle_incidence.size
        else:
            num_signal = angle_incidence.shape[1]

        incidence_signal = signal.gen(
            n=num_signal, nsamples=nsamples, amp=amp, use_cache=use_cache
        )

        if (
            hasattr(signal, "_multipath_enabled")
            and signal._multipath_enabled
            # multipath DOA is only supported for RandomFreqSignal
            and isinstance(signal, RandomFreqSignal)
        ):
            angle_incidence = self._get_multi_path_doa(
                angle_incidence, signal._num_paths
            )
            # used to get the multipath DOAs
            self._doa = angle_incidence

        manifold_matrix = self._steering_vector_with_error(
            signal.frequency, angle_incidence, unit="rad"
        )

        received = manifold_matrix @ incidence_signal

        if snr is not None:
            received = self._add_noise(received, snr)

        return received

    def _gen_broadband(
        self,
        signal: BroadSignal,
        snr,
        nsamples,
        angle_incidence,
        amp,
        use_cache=False,
        calc_method: Literal["delay", "fft"] = "delay",
        **kwargs,
    ):
        assert calc_method in ["fft", "delay"], "Invalid calculation method"

        if calc_method == "fft":
            return self._gen_broadband_fft(
                signal=signal,
                snr=snr,
                nsamples=nsamples,
                angle_incidence=angle_incidence,
                amp=amp,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            return self._gen_broadband_delay(
                signal=signal,
                snr=snr,
                nsamples=nsamples,
                angle_incidence=angle_incidence,
                amp=amp,
                use_cache=use_cache,
                **kwargs,
            )

    def _gen_broadband_fft(
        self,
        signal: BroadSignal,
        snr,
        nsamples,
        angle_incidence,
        amp,
        use_cache=False,
        **kwargs,
    ):
        """Generate broadband received signal using FFT model

        `azimuth` and `elevation` are already in radians
        """
        if angle_incidence.ndim == 1:
            num_signal = angle_incidence.size
        else:
            num_signal = angle_incidence.shape[1]

        num_antennas = self._element_position.shape[0]

        incidence_signal = signal.gen(
            n=num_signal,
            nsamples=nsamples,
            amp=amp,
            use_cache=use_cache,
            delay=None,
            **kwargs,
        )

        # generate array signal in frequency domain
        signal_fre_domain = np.fft.fft(incidence_signal, axis=1)

        received_fre_domain = np.zeros(
            (num_antennas, nsamples), dtype=np.complex128
        )
        fre_points = np.fft.fftfreq(nsamples, 1 / signal.fs)
        for i, fre in enumerate(fre_points):
            manifold_fre = self._steering_vector_with_error(
                fre, angle_incidence, unit="rad"
            )

            # calculate array received signal at every frequency point
            received_fre_domain[:, i] = manifold_fre @ signal_fre_domain[:, i]

        received = np.fft.ifft(received_fre_domain, axis=1)

        if snr is not None:
            received = self._add_noise(received, snr)

        return received

    def _gen_broadband_delay(
        self,
        signal: BroadSignal,
        snr,
        nsamples,
        angle_incidence,
        amp,
        use_cache=False,
        **kwargs,
    ):
        """Generate broadband received signal by applying delay

        `azimuth` and `elevation` are already in radians
        """
        if angle_incidence.ndim == 1:
            num_signal = angle_incidence.size
        else:
            num_signal = angle_incidence.shape[1]

        if np.squeeze(angle_incidence).ndim == 1 or angle_incidence.size == 1:
            angle_incidence = np.vstack(
                (angle_incidence.reshape(1, -1), np.zeros(angle_incidence.size))
            )

        angle_incidence = np.reshape(angle_incidence, (2, -1))

        # calculate time delay
        cos_cos = np.cos(angle_incidence[0]) * np.cos(angle_incidence[1])
        sin_cos = np.sin(angle_incidence[0]) * np.cos(angle_incidence[1])
        sin_ = np.sin(angle_incidence[1])
        time_delay = -(
            1 / C * self._element_position @ np.vstack((cos_cos, sin_cos, sin_))
        )

        received = np.zeros((self.num_antennas, nsamples), dtype=np.complex128)

        # clear cache if not use cache
        if not use_cache:
            signal.clear_cache()

        # must use cache as the same signal is received by different antennas
        for i in range(self.num_antennas):
            received[i, :] = np.sum(
                signal.gen(
                    n=num_signal,
                    nsamples=nsamples,
                    amp=amp,
                    use_cache=True,
                    delay=time_delay[i, :],
                    **kwargs,
                ),
                axis=0,
            )

        if snr is not None:
            received = self._add_noise(received, snr)

        return received

    def _add_noise(self, signal, snr_db):
        sig_pow = np.mean(np.abs(signal) ** 2, axis=1)
        noise_pow = sig_pow / 10 ** (snr_db / 10)

        noise = (np.sqrt(noise_pow / 2)).reshape(-1, 1) * (
            self._rng.standard_normal(size=signal.shape)
            + 1j * self._rng.standard_normal(size=signal.shape)
        )

        # if spatial correlation matrix is provided, add correlated noise
        if hasattr(self, "_correlation_matrix"):
            # use Cholesky decomposition to generate spatially correlated noise
            matrix_sqrt = np.linalg.cholesky(self._correlation_matrix)
            noise = matrix_sqrt @ noise

        return signal + noise

    def _get_multi_path_doa(self, real_doa, num_paths):
        """Generate multipath DOAs

        Args:
            real_doa: Real DOAs
            num_paths: number of paths
        """
        if real_doa.ndim == 1:
            num_signal = real_doa.size
        else:
            warnings.warn("Only 1D DOA is supported", UserWarning)

        multipath_angles = np.zeros(num_signal * (num_paths + 1))
        multipath_angles[:num_signal] = real_doa

        for i in range(num_paths):
            # angle offsets between -30 and 30 degrees
            angle_offsets = self._rng.uniform(-np.pi / 6, np.pi / 6, num_signal)
            # add offsets to the original angles
            multipath_angles[(i + 1) * num_signal : (i + 2) * num_signal] = (
                angle_offsets + real_doa
            )

        return multipath_angles


class UniformLinearArray(Array):
    def __init__(self, m: int, dd: float, rng=None):
        """Uniform linear array.

        The array is uniformly arranged along the y-axis.

        Args:
            m (int): number of antenna elements
            dd (float): distance between adjacent antennas
            rng (np.random.Generator): random generator used to generator random
        """
        # antenna position in (x, y, z) coordinate system
        element_position_x = np.zeros(m)
        element_position_y = np.arange(m) * dd
        element_position_z = np.zeros(m)

        super().__init__(
            element_position_x, element_position_y, element_position_z, rng
        )

    @override
    def add_position_error(self, error_std=0.3, error_type="uniform"):
        dd = self._ideal_position[1, 1] - self._ideal_position[0, 1]
        # the error is added to the distance between adjacent antennas
        sigma = error_std * dd
        if error_type == "gaussian":
            error = self._rng.normal(0, sigma, self.num_antennas)
        elif error_type == "uniform":
            error = self._rng.uniform(-sigma, sigma, self.num_antennas)
        else:
            raise ValueError("Invalid error type")

        self._element_position[:, 1] = self._ideal_position[:, 1] + error

    @override
    def get_default_coupling_matrix(self, rho=0.6):
        """Add mutual coupling effects to the array

        Args:
            rho (float): amplitude of the mutual coupling
            coupling_matrix (np.ndarray): A square matrix representing mutual
                coupling between array elements. Should be of size
                (num_antennas, num_antennas).
        """
        # reference: Liu, Zhang-Meng, Chenwei Zhang, and Philip S. Yu.
        # “Direction-of-Arrival Estimation Based on Deep Neural Networks
        # With Robustness to Array Imperfections.” IEEE Transactions on
        # Antennas and Propagation 66, no. 12 (December 2018): 7315–27.
        # https://doi.org/10.1109/TAP.2018.2874430.

        coefficient = (rho * np.exp(1j * np.pi / 3)) ** np.arange(
            self.num_antennas
        )
        coefficient[0] = 0
        coupling_matrix = toeplitz(coefficient) + np.eye(self.num_antennas)

        return coupling_matrix

    @override
    def get_default_correlation_matrix(self, rho=0.5):
        # reference: Agrawal, M., and S. Prasad. “A Modified Likelihood
        # Function Approach to DOA Estimation in the Presence of Unknown
        # Spatially Correlated Gaussian Noise Using a Uniform Linear Array.”
        # IEEE Transactions on Signal Processing 48, no. 10 (October 2000):
        # 2743–49. https://doi.org/10.1109/78.869024.
        correlation_matrix = (rho ** np.arange(self.num_antennas)) * np.exp(
            -1j * np.pi / 2 * np.arange(self.num_antennas)
        )
        correlation_matrix = toeplitz(correlation_matrix)

        return correlation_matrix


class UniformCircularArray(Array):
    def __init__(self, m, r, rng=None):
        """Uniform circular array.

        The origin is taken as the center of the circle, and the
        counterclockwise direction is considered as the positive direction.

        Args:
            m (int): Number of antennas.
            r (float): Radius of the circular array.
            rng (optional): Random number generator. Defaults to None.
        """
        self._radius = r

        element_position_x = r * np.cos(2 * np.pi * np.arange(m) / m)
        element_position_y = r * np.sin(2 * np.pi * np.arange(m) / m)
        element_position_z = np.zeros(m)

        super().__init__(
            element_position_x, element_position_y, element_position_z, rng
        )

    @property
    def radius(self):
        return self._radius
