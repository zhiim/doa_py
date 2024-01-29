from abc import ABC, abstractmethod

import numpy as np

C = 3e8  # wave speed


class Array(ABC):
    def __init__(self, element_position, rng=None):
        self._element_position = element_position

        if rng is None:
            self._rng = np.random.default_rng()

    @property
    @abstractmethod
    def num_antennas(self):
        raise NotImplementedError()

    @property
    def array_position(self):
        return self._element_position

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

    @abstractmethod
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
        raise NotImplementedError()

    def received_signal(self, signal, snr, angle_incidence, amp=None,
                        broadband=False, unit="deg"):
        """Generate array received signal based on array signal model

        If `broadband` is set to True, generate array received signal based on
        broadband signal's model.

        Args:
            signal: An instance of the `Signal` class
            snr: Signal-to-noise ratio
            angle_incidence: Incidence angle. If only azimuth is considered,
                `angle_incidence` is a 1xN dimensional matrix; if two dimensions
                are considered, `angle_incidence` is a 2xN dimensional matrix,
                where the first row is the azimuth and the second row is the
                elevation angle.
            amp: The amplitude of each signal, 1d numpy array
            broadband: Whether to generate broadband received signals
            unit: The unit of the angle, `rad` represents radian,
                `deg` represents degree. Defaults to 'deg'.
        """
        # Convert the angle from degree to radians
        angle_incidence = self._unify_unit(angle_incidence, unit)

        if broadband is False:
            received = self._gen_narrowband(signal, snr, angle_incidence, amp)
        else:
            received = self._gen_broadband(signal, snr, angle_incidence, amp)

        return received

    @abstractmethod
    def _gen_narrowband(self, signal, snr, angle_incidence, amp):
        """Generate narrowband received signal

        `azimuth` and `elevation` are already in radians
        """
        raise NotImplementedError()

    @abstractmethod
    def _gen_broadband(self, signal, snr, angle_incidence, amp):
        """Generate broadband received signal

        `azimuth` and `elevation` are already in radians
        """
        raise NotImplementedError()


class UniformLinearArray(Array):
    def __init__(self, m: int, dd: float, rng=None):
        """Uniform linear array.

        Args:
            m (int): number of antenna elements
            dd (float): distance between adjacent antennas
            rng (np.random.Generator): random generator used to generator random
        """
        # array position should be a 2d Mx1 numpy array
        super().__init__(np.arange(m).reshape(-1, 1) * dd, rng)

    @property
    def num_antennas(self):
        return self._element_position.size

    def steering_vector(self, fre, angle_incidence, unit="deg"):
        angle_incidence = self._unify_unit(np.reshape(angle_incidence, (1, -1)),
                                           unit)

        time_delay = 1 / C * self.array_position @ np.sin(angle_incidence)
        steering_vector = np.exp(-1j * 2 * np.pi * fre * time_delay)

        return steering_vector

    def _gen_narrowband(self, signal, snr, angle_incidence, amp):
        """We only consider azimuth when use ULA, so `angle_incidence` should
        be a 1xN array.
        """
        azimuth = angle_incidence.reshape(1, -1)
        num_signal = azimuth.shape[1]

        # calculate the time delay matrix
        matrix_tau = 1 / C * self._element_position @ np.sin(azimuth)
        # calcualte the manifold matrix
        manifold_matrix = np.exp(-1j * 2 * np.pi * signal.frequency *
                                 matrix_tau)

        incidence_signal = signal.gen(n=num_signal, amp=amp)

        received = manifold_matrix @ incidence_signal

        noise = 1 / np.sqrt(10 ** (snr / 10)) * np.mean(np.abs(received)) *\
            1 / np.sqrt(2) * (self._rng.standard_normal(size=received.shape) +
                            1j * self._rng.standard_normal(size=received.shape))
        received = received + noise

        return received

    def _gen_broadband(self, signal, snr, angle_incidence, amp):
        azimuth = angle_incidence.reshape(1, -1)
        num_signal = azimuth.shape[1]
        num_snapshots = signal.nsamples
        num_antennas = self._element_position.size

        incidence_signal = signal.gen(n=num_signal, amp=amp)

        # generate array signal in frequency domain
        signal_fre_domain = np.fft.fft(incidence_signal, axis=1)

        matrix_tau = 1 / C * self._element_position @ np.sin(azimuth)

        received_fre_domain = np.zeros((num_antennas, num_snapshots),
                                       dtype=np.complex_)
        fre_points = np.fft.fftfreq(num_snapshots, 1 / signal.fs)
        for i, fre in enumerate(fre_points):
            manifold_fre = np.exp(-1j * 2 * np.pi * fre * matrix_tau)

            # calculate array received signal at every frequency point
            received_fre_domain[:, i] = manifold_fre @ signal_fre_domain[:, i]

        received = np.fft.ifft(received_fre_domain, axis=1)

        noise = 1 / np.sqrt(10 ** (snr / 10)) * np.mean(np.abs(received)) *\
            1 / np.sqrt(2) * (self._rng.standard_normal(size=received.shape) +
                            1j * self._rng.standard_normal(size=received.shape))
        received = received + noise

        return received


class UniformCircularArray(Array):
    def __init__(self, element_position, rng=None):
        super().__init__(element_position, rng)
