import numpy as np
from ..music import MUSIC
from .wideband_core import divide_wideband_into_sub, get_estimates_from_sp

C = 3e8  # wave speed

class CSSM(MUSIC):
    """Coherent Signal Subspace Method (CSSM) for wideband DOA estimation.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
    """
    def __init__(self, array, search_grid, **kwargs):
        wavelength = None
        super().__init__(array, wavelength, search_grid, enable_caching=False,
                         **kwargs)

    def _spatial_spectrum(self, signal, fs, f_start, f_end, pre_estimate,
                          n_fft, k):
        """Get spatial spectrum using CSSM"""
        # 获取阵元位置，并将其转换为M*1维向量
        array_location = self._array.actual_element_locations
        # reshape to a 1*N array
        pre_estimate = pre_estimate

        # 选取频带的中心频率作为参考频点
        f_reference = (f_start + f_end) / 2
        # 计算参考频点下，预估计角度对应的流型矩阵
        matrix_a_ref = np.exp(1j * 2 * np.pi * f_reference / C *\
                              np.outer(array_location, np.sin(pre_estimate)))

        # divide wideband signal into different frequency points
        signal_subs, freq_bins = divide_wideband_into_sub(signal=signal,
                                                          n_fft=n_fft, fs=fs,
                                                          f_start=f_start,
                                                          f_end=f_end)
        num_group = signal_subs.shape[2]  # 每个频点对应的FFT的次数

        matrix_r = np.zeros((self._array.size, self._array.size),
                            dtype=np.complex_)  # 协方差矩阵
        for i, freq in enumerate(freq_bins):
            # 每个频点下，角度预估值对应的流型矩阵
            matrix_a_f = np.exp(1j * 2 * np.pi * freq / C *\
                                np.outer(array_location, np.sin(pre_estimate)))
            matrix_q = matrix_a_f @ matrix_a_ref.conj().T
            # 对matrix_q进行奇异值分解
            matrix_u, _, matrix_vh = np.linalg.svd(matrix_q)
            # RSS法构造最佳聚焦矩阵
            matrix_t_f = matrix_vh.conj().T @ matrix_u.conj().T
            # 取出每个频点对应的频域接收信号
            matrix_x_f = signal_subs[:, i, :]
            matrix_r += matrix_t_f @\
                        (matrix_x_f @ matrix_x_f.conj().T / num_group)\
                        @ matrix_t_f.conj().T
        # 对不同频点下，聚焦后的协方差矩阵做平均得到最终的协方差矩阵
        matrix_r = matrix_r / freq_bins.size

        self._wavelength = C / f_reference
        # get spatial specturm using MUSIC algorithm
        spatial_spectrum = super()._spatial_spectrum(matrix_r=matrix_r, k=k)
        return spatial_spectrum

    def estimate(self, signal, fs, f_start, f_end, pre_estimate, n_fft, k,
                 return_spectrum=True):
        """Get DOA estimation using CSSM.

        Args:
            signal (np.array): sampled wideband signal.
            fs (float): sampling frequency.
            f_start (float): start frequency of wideband signal.
            f_end (float): end frequency of wideband signal.
            pre_estimate (np.array): a 1D array consists pre-estimate of DOA of
                every incident signal.
            n_fft (int): number of points of FFT.
            k (int): number of sources.
            return_spectrum (bool, optional): return spatial spectrum or not.
                Defaults to True.

        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): A boolean indicating if the desired
              number of sources are found. This flag does **not** guarantee that
              the estimated source locations are correct. The estimated source
              locations may be completely wrong!
              If resolved is False, both ``estimates`` and ``spectrum`` will be
              ``None``.
            * estimates (:class:`~doatools.model.sources.SourcePlacement`):
              A :class:`~doatools.model.sources.SourcePlacement` instance of the
              same type as the one used in the search grid, represeting the
              estimated source locations. Will be ``None`` if resolved is
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same
              shape of the specified search grid, consisting of values evaluated
              at the grid points. Only present if ``return_spectrum`` is
              ``True``.
        """
        sp = self._spatial_spectrum(signal=signal, fs=fs, f_start=f_start,
                                    f_end=f_end, pre_estimate=pre_estimate,
                                    n_fft=n_fft, k=k)
        return get_estimates_from_sp(sp=sp, k=k, search_grid=self._search_grid,
                                     peak_finder=self._peak_finder,
                                     return_spectrum=return_spectrum)
