import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def plot_spatial_spectrum(spectrum, ground_truth, angle_grids,
                          peak_threshold=0.2, x_label="Angle",
                          y_label="Spectrum"):
    """Plot spatial spectrum

    Args:
        spectrum: Spatial spectrum estimated by the algorithm
        ground_truth: True incident angles
        angle_grids: Angle grids corresponding to the spatial spectrum
        peak_threshold: Threshold used to find peaks in normalized spectrum
        x_label: x-axis label
        y_label: y-axis label
    """
    spectrum = spectrum / np.max(spectrum)
    # find peaks and peak heights
    peaks_idx, heights = find_peaks(spectrum,
                                    height=peak_threshold)
    angles = angle_grids[peaks_idx]
    heights = heights["peak_heights"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # set ticks
    grids_min = angle_grids[0]
    grids_max = angle_grids[-1]
    major_space = (grids_max - grids_min + 1) / 6
    minor_space = major_space / 5
    ax.set_xlim(grids_min, grids_max)
    ax.xaxis.set_major_locator(plt.MultipleLocator(major_space))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_space))

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
                         x_label="Angle", y_label="Spectrum"):
    """Display estimated angle values

    Args:
        estimates: Angle estimates
        ground_truth: True incident angles
        ticks_min (int, optional): Minimum value for x-axis ticks.
            Defaults to -90.
        ticks_max (int, optional): Maximum value for x-axis ticks.
            Defaults to 90.
        x_label (str, optional): x-axis label. Defaults to "Angle".
        y_label (str, optional): y-axis label. Defaults to "Spetrum".
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # set ticks
    major_space = (ticks_max - ticks_min) / 6
    minor_space = major_space / 5
    ax.set_xlim(ticks_min, ticks_max)
    ax.xaxis.set_major_locator(plt.MultipleLocator(major_space))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_space))

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
