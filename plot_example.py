"""
Plot a terrain model and a polar plot side by side.
"""

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import mpl_toolkits.mplot3d  # noqa: F401

plt.style.use("dark_background")


def generate_polar_plot():
    """Generate an example polar slice plot."""
    # Compute pie slices
    n_slices = 20
    theta = np.linspace(0.0, 2 * np.pi, n_slices, endpoint=False)
    radii = 10 * np.random.rand(n_slices)
    width = np.pi / 4 * np.random.rand(n_slices)

    fig = plt.gcf()
    fig.patch.set_facecolor("#395979")

    ax1 = plt.subplot(1, 2, 2, projection="polar")
    ax1.set_facecolor("#395979")

    bars = ax1.bar(theta, radii, width=width, bottom=0.0)

    # Use custom colors and opacity
    for radius, plot_bar in zip(radii, bars):
        plot_bar.set_facecolor(plt.cm.viridis(radius / 10.0))
        plot_bar.set_alpha(0.5)


def generate_dem_plot():
    """Generate a 3D representation of a terrain DEM."""
    dem_path = "jacksboro_fault_dem.npz"

    with np.load(dem_path) as dem:
        z_data = dem["elevation"]
        nrows, ncols = z_data.shape

        x_data = np.linspace(dem["xmin"], dem["xmax"], ncols)
        y_data = np.linspace(dem["ymin"], dem["ymax"], nrows)
        x_data, y_data = np.meshgrid(x_data, y_data)

    region = np.s_[5:50, 5:50]
    x_region, y_region, z_region = (
        x_data[region],
        y_data[region],
        z_data[region],
    )

    axes = plt.subplot(1, 2, 1, projection="3d")
    axes.set_facecolor("#395979")

    plt.locator_params(axis="y", nbins=6)
    plt.locator_params(axis="x", nbins=6)

    light_source = matplotlib.colors.LightSource(270, 45)

    rgb_map = light_source.shade(z_region, cmap=matplotlib.cm.gist_earth)

    axes.plot_surface(
        x_region,
        y_region,
        z_region,
        facecolors=rgb_map,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def main():
    plt.figure(figsize=(10, 5))

    generate_dem_plot()
    generate_polar_plot()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
