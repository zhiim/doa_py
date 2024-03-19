import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from skimage.feature import peak_local_max


def plot_spatial_spectrum(
    spectrum,
    ground_truth,
    angle_grids,
    num_signal,
    x_label="Angle",
    y_label="Spectrum",
):
    """Plot spatial spectrum

    Args:
        spectrum: Spatial spectrum estimated by the algorithm
        ground_truth: True incident angles
        angle_grids: Angle grids corresponding to the spatial spectrum
        num_signal: Number of signals
        x_label: x-axis label
        y_label: y-axis label
    """
    spectrum = spectrum / np.max(spectrum)
    # find peaks and peak heights
    peaks_idx, heights = find_peaks(spectrum, height=0)

    idx = heights["peak_heights"].argsort()[-num_signal:]
    peaks_idx = peaks_idx[idx]
    heights = heights["peak_heights"][idx]

    angles = angle_grids[peaks_idx]

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
    ax.set_yscale("log")

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


def plot_estimated_value(
    estimates,
    ground_truth,
    ticks_min=-90,
    ticks_max=90,
    x_label="Angle",
    y_label="Spectrum",
):
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


def plot_spatial_spectrum_2d(
    spectrum,
    ground_truth,
    azimuth_grids,
    elevation_grids,
    x_label="Elevation",
    y_label="Azimuth",
    z_label="Spectrum",
):
    """Plot 2D spatial spectrum

    Args:
        spectrum: Spatial spectrum estimated by the algorithm
        ground_truth: True incident angles
        azimuth_grids : Azimuth grids corresponding to the spatial spectrum
        elevation_grids : Elevation grids corresponding to the spatial spectrum
        x_label: x-axis label
        y_label: y-axis label
        z_label : x-axis label. Defaults to "Spectrum".
    """
    x, y = np.meshgrid(elevation_grids, azimuth_grids)
    spectrum = spectrum / spectrum.max()
    # Find the peaks in the surface
    peaks = peak_local_max(spectrum, num_peaks=ground_truth.shape[1])
    spectrum = np.log(spectrum + 1e-10)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # plot spectrum
    surf = ax.plot_surface(x, y, spectrum, cmap="viridis", antialiased=True)
    # Plot the peaks on the surface
    for peak in peaks:
        peak_dot = ax.scatter(
            x[peak[0], peak[1]],
            y[peak[0], peak[1]],
            spectrum[peak[0], peak[1]],
            c="r",
            marker="x",
        )
        ax.text(
            x[peak[0], peak[1]],
            y[peak[0], peak[1]],
            spectrum[peak[0], peak[1]],
            "({}, {})".format(x[peak[0], peak[1]], y[peak[0], peak[1]]),
        )
    # plot ground truth
    truth_lines = ax.stem(
        ground_truth[1],
        ground_truth[0],
        np.ones_like(ground_truth[0]),
        bottom=spectrum.min(),
        linefmt="g--",
        markerfmt=" ",
        basefmt=" ",
    )

    ax.legend(
        [surf, truth_lines, peak_dot], ["Spectrum", "Estimated", "Ground Truth"]
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    plt.show()


def plot_estimated_value_2d(
    estimated_azimuth,
    estimated_elevation,
    ground_truth,
    unit="deg",
    x_label="Angle",
    y_label="Spectrum",
):
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
    if unit == "deg":
        estimated_azimuth = estimated_azimuth / 180 * np.pi
        ground_truth = ground_truth.astype(float)
        ground_truth[0] = ground_truth[0] / 180 * np.pi

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    ax.scatter(ground_truth[0], ground_truth[1], marker="o", color="g")
    ax.scatter(estimated_azimuth, estimated_elevation, marker="x", color="r")

    ax.set_rlabel_position(90)

    for i in range(len(estimated_azimuth)):
        ax.annotate(
            "({:.2f}, {:.2f})".format(
                estimated_azimuth[i] / np.pi * 180, estimated_elevation[i]
            ),
            (estimated_azimuth[i], estimated_elevation[i]),
        )

    ax.set_xticks(np.arange(0, 2 * np.pi, step=np.pi / 6))
    ax.set_rlim([0, 90])
    ax.set_yticks(np.arange(0, 90, step=15))

    ax.legend(["Ground Truth", "Estimated"])

    plt.show()
