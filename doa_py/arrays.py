from abc import ABC
from typing import Literal

import numpy as np
from scipy.linalg import toeplitz
from typing_extensions import override

from .signals import BroadSignal, NarrowSignal

C = 3e8  # wave speed


class Array(ABC):
    def __init__(
        self,
        element_position_x: np.ndarray,
        element_position_y: np.ndarray,
        element_position_z: np.ndarray,
        rng: np.random.Generator | None = None,
    ):
        """element position should be defined in 3D (x, y, z) coordinate
        system"""
        self._element_position = np.vstack(
            (element_position_x, element_position_y, element_position_z)
        ).T

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    @property
    def num_antennas(self):
        return self._element_position.shape[0]

    @property
    def array_position(self):
        return self._element_position

    def set_rng(self, rng: np.random.Generator):
        """Setting random number generator

        Args:
            rng (np.random.Generator): random generator used to generator random
        """
        self._rng = rng

    def _unify_unit(self, variable, unit):
        """Convert variable to radian"""
        if unit == "deg":
            variable = variable / 180 * np.pi

        return variable

    def steering_vector(
        self, fre: float, angle_incidence: np.ndarray, unit: str = "deg"
    ) -> np.ndarray:
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

        time_delay = (
            1 / C * self._element_position @ np.vstack((cos_cos, sin_cos, sin_))
        )
        steering_vector = np.exp(-1j * 2 * np.pi * fre * time_delay)

        return steering_vector

    def received_signal(
        self,
        signal: NarrowSignal,
        angle_incidence: np.ndarray,
        snr: int | float | None = None,
        nsamples: int = 100,
        amp: float = None,
        unit: str = "deg",
        use_cache: bool = False,
    ) -> np.ndarray:
        """Generate array received signal based on array signal model, only
        narrowband signal is supported.

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
        """
        if not isinstance(signal, NarrowSignal):
            raise ValueError(
                "Narrowband signal is required for this method. Use "
                "`received_signal_broad` method for "
                "`{signal.__class__.__name__}` instead."
            )

        # Convert the angle from degree to radians
        angle_incidence = self._unify_unit(angle_incidence, unit)

        if angle_incidence.ndim == 1:
            num_signal = angle_incidence.size
        else:
            num_signal = angle_incidence.shape[1]

        incidence_signal = signal.gen(
            n=num_signal, nsamples=nsamples, amp=amp, use_cache=use_cache
        )

        manifold_matrix = self.steering_vector(
            signal.frequency, angle_incidence, unit="rad"
        )

        received = manifold_matrix @ incidence_signal

        if snr is not None:
            received += self._add_noise(received, snr)

        return received

    def received_signal_broad(
        self,
        signal: BroadSignal,
        angle_incidence: np.ndarray,
        snr: int | float | None = None,
        nsamples=100,
        amp=None,
        unit="deg",
        use_cache=False,
        calc_method: Literal["delay", "fft"] = "delay",
        **kwargs,
    ):
        """Generate array received signal based on array signal model, only
        broadband signal is supported.

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
            **kwargs: Additional parameters passed to the signal generation
        """
        if not isinstance(signal, BroadSignal):
            raise ValueError(
                "Broadband signal is required for this method. Use "
                "`received_signal_broad` method for "
                "`{signal.__class__.__name__}` instead."
            )

        # Convert the angle from degree to radians
        angle_incidence = self._unify_unit(angle_incidence, unit)

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

        num_antennas = self.num_antennas

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
            manifold_fre = self.steering_vector(
                fre, angle_incidence, unit="rad"
            )

            # calculate array received signal at every frequency point
            received_fre_domain[:, i] = manifold_fre @ signal_fre_domain[:, i]

        received = np.fft.ifft(received_fre_domain, axis=1)

        if snr is not None:
            received += self._add_noise(received, snr)

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
            received += self._add_noise(received, snr)

        return received

    def _add_noise(self, signal, snr_db):
        sig_pow = np.mean(np.abs(signal) ** 2, axis=1)
        noise_pow = sig_pow / 10 ** (snr_db / 10)

        noise = (np.sqrt(noise_pow / 2)).reshape(-1, 1) * (
            self._rng.standard_normal(size=signal.shape)
            + 1j * self._rng.standard_normal(size=signal.shape)
        )

        return noise


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

        # used to store the real position with error
        self._real_position = self._element_position.copy()

    def add_position_error(
        self,
        pos_error: np.ndarray,
    ):
        """Add position error to the array

        Args:
            pos_error (np.ndarray): An array of shape (num_antennas,)
                representing the position error in x, y, z directions for each
                antenna
        """
        if pos_error.shape != (self.num_antennas,):
            raise ValueError(
                f"Position error shape {pos_error.shape} does not match "
                f"the position of antennas ({self.num_antennas},)"
            )
        self._real_position[:, 1] = self._element_position[:, 1] + pos_error

    def add_position_error_default(
        self,
        error_std: float = 0.3,
        error_type: Literal["uniform", "gaussian"] = "uniform",
    ):
        """Generate position error randomly to the array. It can be used when
        no specific position error is provided.

        Args:
            error_std (float): Standard deviation of the position error
            error_type (str): Type of the error, `gaussian` or `uniform`
        """
        dd = self._element_position[1, 1] - self._element_position[0, 1]
        # the error is added to the distance between adjacent antennas
        sigma = error_std * dd
        if error_type == "gaussian":
            error = self._rng.normal(0, sigma, self.num_antennas)
        elif error_type == "uniform":
            error = self._rng.uniform(-sigma, sigma, self.num_antennas)

        self.add_position_error(error)

    def add_mutual_coupling(self, coupling_matrix: np.ndarray):
        """Add mutual coupling effects to the array

        Args:
            coupling_matrix (np.ndarray): A square matrix representing mutual
                coupling between array elements. Should be of size
                (num_antennas, num_antennas).
        """
        if coupling_matrix.shape != (self.num_antennas, self.num_antennas):
            raise ValueError(
                f"Coupling matrix shape {coupling_matrix.shape} does not match "
                f"({self.num_antennas}, {self.num_antennas})"
            )

        self._coupling_matrix = coupling_matrix

    def add_mutual_coupling_default(self, rho=0.6):
        """Add mutual coupling effects to the array using default generated
        coupling matrix.

        Args:
            rho (float): amplitude of the mutual coupling
            coupling_matrix (np.ndarray): A square matrix representing mutual
                coupling between array elements. Should be of size
                (num_antennas, num_antennas).

        Reference:
            Liu, Zhang-Meng, Chenwei Zhang, and Philip S. Yu.
            “Direction-of-Arrival Estimation Based on Deep Neural Networks
            With Robustness to Array Imperfections.”
            IEEE Transactions on Antennas and Propagation 66, no. 12
            (December 2018): 7315–27.
            https://doi.org/10.1109/TAP.2018.2874430.
        """
        coefficient = (rho * np.exp(1j * np.pi / 3)) ** np.arange(
            self.num_antennas
        )
        coefficient[0] = 0
        coupling_matrix = toeplitz(coefficient) + np.eye(self.num_antennas)

        self.add_mutual_coupling(coupling_matrix)

    def add_gain_phase_error(
        self, gain_error: np.ndarray, phase_error: np.ndarray
    ):
        """Add gain and phase error to the array

        Args:
            gain_error (np.ndarray): An array of shape (num_antennas,)
                representing the gain error in x, y, z directions for each
                antenna
            phase_error (np.ndarray): An array of shape (num_antennas,)
                representing the phase error in x, y, z directions for each
                antenna
        """
        if gain_error.shape != (self.num_antennas,):
            raise ValueError(
                f"Gain error shape {gain_error.shape} does not match "
                f"the number of antennas ({self.num_antennas},)"
            )
        if phase_error.shape != (self.num_antennas,):
            raise ValueError(
                f"Phase error shape {phase_error.shape} does not match "
                f"the number of antennas ({self.num_antennas},)"
            )

        self._gain_phase_error_matrix = np.diag(
            gain_error * np.exp(1j * phase_error)
        )

    def add_gain_phase_error_default(
        self, gain_error_std: float = 0.1, phase_error_amp: float = 2 * np.pi
    ):
        """Generate gain and phase error randomly to the array. It can be used
        when no specific gain and phase error is provided.

        Args:
            gain_error_std (float): Standard deviation of the gain error in
                gaussian distribution
            phase_error_amp (float): Amplitude of the phase error (max expected
                value)
        """
        gain_error = self._rng.normal(
            loc=1.0, scale=gain_error_std, size=self.num_antennas
        )
        phase_error = (phase_error_amp / (2 * np.pi)) * self._rng.uniform(
            low=0, high=2 * np.pi, size=self.num_antennas
        )

        self.add_gain_phase_error(gain_error, phase_error)

    def add_correlatted_noise(self, correlation_matrix=np.ndarray):
        """Add spatial correlation matrix to the array, which is used to
        generate spatially correlated noise.

        If this method is not called, use the spatially and temporally
        uncorrelated noise.

        Args:
            correlation_matrix (np.ndarray): A square matrix representing
                spatial correlation between array elements. Should be of size
                (num_antennas, num_antennas). Defaults to None.
        """
        if correlation_matrix.shape != (self.num_antennas, self.num_antennas):
            raise ValueError(
                f"Correlation matrix shape {correlation_matrix.shape} does not "
                f"match the number of antennas {self.num_antennas}"
            )

        self._correlation_matrix = correlation_matrix

    def add_correlatted_noise_default(self, rho=0.5):
        """Add default generated spatial correlation matrix to the array, which
        is used to generate spatially correlated noise.

        Args:
            rho (float): amplitude of the correlation matrix

        Reference:
            Agrawal, M., and S. Prasad.
            “A Modified Likelihood Function Approach to DOA Estimation in the
            Presence of Unknown Spatially Correlated Gaussian Noise Using a
            Uniform Linear Array.”
            IEEE Transactions on Signal Processing 48, no. 10 (October 2000):
            2743–49.
            https://doi.org/10.1109/78.869024.
        """
        correlation_matrix = (rho ** np.arange(self.num_antennas)) * np.exp(
            -1j * np.pi / 2 * np.arange(self.num_antennas)
        )
        correlation_matrix = toeplitz(correlation_matrix)

        self.add_correlatted_noise(correlation_matrix)

    @override
    def _add_noise(self, signal, snr_db):
        noise = super()._add_noise(signal=signal, snr_db=snr_db)

        # if spatial correlation matrix is provided, add correlated noise
        if hasattr(self, "_correlation_matrix"):
            # use Cholesky decomposition to generate spatially correlated noise
            matrix_sqrt = np.linalg.cholesky(self._correlation_matrix)
            noise = matrix_sqrt @ noise

        return noise

    @override
    def received_signal(
        self,
        signal: NarrowSignal,
        angle_incidence: np.ndarray,
        snr: int | float | None = None,
        nsamples: int = 100,
        amp: float = None,
        unit: str = "deg",
        use_cache: bool = False,
    ) -> np.ndarray:
        element_position_backup = self._element_position.copy()
        self._element_position = self._real_position  # consider position error
        received = super().received_signal(
            signal=signal,
            angle_incidence=angle_incidence,
            snr=None,  # noise added later to add other errors
            nsamples=nsamples,
            amp=amp,
            unit=unit,
            use_cache=use_cache,
        )
        self._element_position = element_position_backup  # restore position

        # add mutual coupling effect if provided
        if hasattr(self, "_coupling_matrix"):
            received = self._coupling_matrix @ received

        # add gain and phase error if provided
        if hasattr(self, "_gain_phase_error_matrix"):
            received = self._gain_phase_error_matrix @ received

        # add noise
        if snr is not None:
            received += self._add_noise(received, snr)

        return received

    @override
    def received_signal_broad(
        self,
        signal: BroadSignal,
        angle_incidence: np.ndarray,
        snr: int | float | None = None,
        nsamples=100,
        amp=None,
        unit="deg",
        use_cache=False,
        calc_method: Literal["delay", "fft"] = "delay",
        **kwargs,
    ):
        element_position_backup = self._element_position.copy()
        self._element_position = self._real_position  # consider position error
        received = super().received_signal_broad(
            signal=signal,
            angle_incidence=angle_incidence,
            snr=None,  # noise added later to add other errors
            nsamples=nsamples,
            amp=amp,
            unit=unit,
            use_cache=use_cache,
            calc_method=calc_method,
            **kwargs,
        )
        self._element_position = element_position_backup  # restore position

        # add mutual coupling effect if provided
        if hasattr(self, "_coupling_matrix"):
            received = self._coupling_matrix @ received

        # add gain and phase error if provided
        if hasattr(self, "_gain_phase_error_matrix"):
            received = self._gain_phase_error_matrix @ received

        # add noise
        if snr is not None:
            received += self._add_noise(received, snr)

        return received


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
