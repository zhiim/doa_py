import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def plot_spatial_specturm(spectrum, ground_truth, angle_grids,
                          peak_threshold=0.5, x_label="Angle",
                          y_label="Spetrum"):
    """绘制空间谱

    Args:
        spectrum: 算法估计的空间谱
        ground_truth: 真是入射角度
        angle_grids: 空间谱对应的角度网格点
        peak_threshold: 用于寻找峰值的阈值(相对于最大值的比例)
        x_label: x轴标度
        y_label: y轴标度
    """
    spectrum = spectrum / np.max(spectrum)
    # find peaks and peaks' height
    peaks_idx, heights = find_peaks(spectrum,
                                    height=np.max(spectrum) * peak_threshold)
    angles = angle_grids[peaks_idx]
    heights = heights["peak_heights"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # set ticks
    grids_min = angle_grids[0]
    grids_max = angle_grids[-1]
    major_space = (grids_max - grids_min + 1) / 6
    minor_space = major_space / 5
    major_ticks = np.arange(grids_min, grids_max, major_space)
    minor_ticks = np.arange(grids_min, grids_max, minor_space)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    # plot spectrum
    ax.plot(angle_grids, spectrum)
    ax.set_yscale('log')

    # plot peaks
    ax.scatter(angles, heights, color="red", marker="x")
    for i, angle in enumerate(angles):
        ax.annotate(angle, xy=(angle, heights[i]))

    # ground truth
    for angle in ground_truth:
        ax.axvline(x=angle, color="green", linestyle="--")

    # set labels
    ax.legend(["Spectrum", "Estimated", "Ground Truth"])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()


def plot_estimated_value(estimates, ground_truth, ticks_min=-90, ticks_max=90,
                         x_label="Angle", y_label="Spetrum"):
    """展示估计的角度值

    Args:
        estimates: 角度估计值
        ground_truth: 真实入射角
        ticks_min (int, optional): x轴刻度的最小值. Defaults to -90.
        ticks_max (int, optional): x轴刻度的最大值. Defaults to 90.
        x_label (str, optional): x轴标度. Defaults to "Angle".
        y_label (str, optional): y轴标度. Defaults to "Spetrum".
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # set ticks
    major_space = (ticks_min - ticks_max + 1) / 6
    minor_space = major_space / 5
    major_ticks = np.arange(ticks_min, ticks_max, major_space)
    minor_ticks = np.arange(ticks_min, ticks_max, minor_space)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    # ground truth
    for angle in ground_truth:
        truth_line = ax.axvline(x=angle, color="c", linestyle="--")

    # plot estimates
    for angle in estimates:
        estimate_line = ax.axvline(x=angle, color="r", linestyle="--")

    # set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # set legend
    ax.legend([truth_line, estimate_line], ["Ground Truth", "Estimated"])

    plt.show()
