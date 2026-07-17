"""
Demo figures for Spyder: paired and single matplotlib plot examples.
"""

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import mpl_toolkits.mplot3d

plt.style.use("dark_background")


def generate_polar_plot(ax, face_color = "#395979"):
    """Generate an example polar slice plot."""
    n_slices = 20
    theta = np.linspace(0.0, 2 * np.pi, n_slices, endpoint=False)
    radii = 10 * np.random.rand(n_slices)
    width = np.pi / 4 * np.random.rand(n_slices)

    ax.set_facecolor(face_color)

    bars = ax.bar(theta, radii, width=width, bottom=0.0)

    for radius, plot_bar in zip(radii, bars):
        plot_bar.set_facecolor(plt.cm.viridis(radius / 10.0))
        plot_bar.set_alpha(0.5)

    ax.set_title("Polar Plot")


def generate_dem_plot(ax, face_color = "#395979"):
    """Generate a 3D representation of a terrain DEM."""
    dem_path = "jacksboro_fault_dem.npz"

    with np.load(dem_path) as dem:
        z_data = dem["elevation"]
        nrows, ncols = z_data.shape

        x_data = np.linspace(dem["xmin"], dem["xmax"], ncols)
        y_data = np.linspace(dem["ymin"], dem["ymax"], nrows)
        x_data, y_data = np.meshgrid(x_data, y_data)

    region = np.s_[5:50, 5:50]
    x_region = x_data[region]
    y_region = y_data[region]
    z_region = z_data[region]

    ax.set_facecolor(face_color)

    light_source = matplotlib.colors.LightSource(270, 45)

    rgb_map = light_source.shade(
        z_region,
        cmap=matplotlib.cm.gist_earth,
    )

    ax.plot_surface(
        x_region,
        y_region,
        z_region,
        facecolors=rgb_map,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax.set_title("Terrain DEM")


def generate_torus_plot(ax, face_color = "#395979"):
    """Generate a colored 3D torus."""
    R = 3.0  # Major radius
    r = 1.0  # Minor radius

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 50)

    u, v = np.meshgrid(u, v)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    norm = matplotlib.colors.Normalize(
        vmin=z.min(),
        vmax=z.max(),
    )

    facecolors = matplotlib.cm.plasma(norm(z))

    ax.set_facecolor(face_color)

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Parametric Torus")


def generate_heatmap(ax):
    data = np.random.rand(20, 20)

    image = ax.imshow(
        data,
        cmap="viridis",
        interpolation="nearest",
    )

    ax.set_title("Heatmap")
    plt.colorbar(image, ax=ax)


def generate_scatter3d(ax, face_color = "#395979"):
    """Generate a random 3D scatter plot."""
    n = 300

    x = np.random.randn(n)
    y = np.random.randn(n)
    z = np.random.randn(n)

    colors = np.sqrt(x**2 + y**2 + z**2)

    ax.set_facecolor(face_color)

    ax.scatter(
        x,
        y,
        z,
        c=colors,
        cmap="viridis",
        s=20,
    )

    ax.set_title("3D Scatter")


def generate_surface_plot(ax, face_color = "#395979"):
    """Generate a mathematical surface."""

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    x, y = np.meshgrid(x, y)

    z = np.sin(x) * np.cos(y)

    ax.set_facecolor(face_color)

    ax.plot_surface(
        x,
        y,
        z,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
    )

    ax.set_title("Surface Plot")


def generate_trisurface_plot(ax, face_color = "#395979"):
    """Generate a triangulated surface."""

    n = 1000
    theta = 2 * np.pi * np.random.rand(n)
    r = np.random.rand(n)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sin(8 * r)

    ax.set_facecolor(face_color)

    ax.plot_trisurf(
        x,
        y,
        z,
        cmap="viridis",
        linewidth=0.2,
    )

    ax.set_title("Triangulated Surface")


def generate_contour3d_plot(ax, face_color = "#395979"):
    """Generate a 3D contour plot."""

    x = np.linspace(-3, 3, 80)
    y = np.linspace(-3, 3, 80)
    x, y = np.meshgrid(x, y)

    z = np.sin(x) * np.cos(y)

    ax.set_facecolor(face_color)

    ax.contour3D(
        x,
        y,
        z,
        50,
        cmap="plasma",
    )

    ax.set_title("3D Contours")
    

def generate_helix_plot(ax, face_color = "#395979"):
    """Generate a 3D helix."""

    t = np.linspace(0, 20 * np.pi, 1000)

    x = np.cos(t)
    y = np.sin(t)
    z = t

    ax.set_facecolor(face_color)

    ax.plot(
        x,
        y,
        z,
        color="gold",
        linewidth=2,
    )

    ax.set_title("3D Helix")


def generate_wireframe_plot(ax, face_color = "#395979"):
    """Generate a 3D wireframe surface."""
    x = np.linspace(-2, 2, 40)
    y = np.linspace(-2, 2, 40)
    x, y = np.meshgrid(x, y)
    z = np.cos(np.sqrt(x**2 + y**2))

    ax.set_facecolor(face_color)
    ax.plot_wireframe(x, y, z, color="cyan", linewidth=0.6)
    ax.set_title("Wireframe")


def generate_bar3d_plot(ax, face_color = "#395979"):
    """Generate a 3D bar chart."""
    n = 5
    x = np.arange(n)
    y = np.arange(n)
    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()
    z = np.zeros_like(x)
    dx = dy = 0.6
    dz = np.random.rand(n * n) * 4 + 1

    ax.set_facecolor(face_color)
    ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=plt.cm.plasma(dz / dz.max()))
    ax.set_title("3D Bars")


def generate_quiver_plot(ax, face_color = "#395979"):
    """Generate a 2D vector field."""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    x, y = np.meshgrid(x, y)
    u = -y
    v = x

    ax.set_facecolor(face_color)
    ax.quiver(x, y, u, v, color="gold")
    ax.set_aspect("equal")
    ax.set_title("Quiver Field")


def generate_contourf_plot(ax, face_color = "#395979"):
    """Generate a filled contour plot."""
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    x, y = np.meshgrid(x, y)
    z = np.sin(x) * np.cos(y)

    ax.set_facecolor(face_color)
    contour = ax.contourf(x, y, z, levels=20, cmap="viridis")
    plt.colorbar(contour, ax=ax)
    ax.set_title("Filled Contours")


def generate_sphere_plot(ax, face_color = "#395979"):
    """Generate a parametric sphere."""
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    u, v = np.meshgrid(u, v)

    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    ax.set_facecolor(face_color)
    ax.plot_surface(x, y, z, cmap="coolwarm", linewidth=0, antialiased=True)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Sphere")


def generate_lorenz_plot(ax, face_color = "#395979"):
    """Generate a Lorenz attractor trajectory."""
    dt = 0.01
    steps = 5000
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    points = np.empty((steps, 3))
    points[0] = (0.0, 1.0, 1.05)

    for i in range(steps - 1):
        x, y, z = points[i]
        points[i + 1] = (
            x + sigma * (y - x) * dt,
            y + (x * (rho - z) - y) * dt,
            z + (x * y - beta * z) * dt,
        )

    ax.set_facecolor(face_color)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="lime", linewidth=0.5)
    ax.set_title("Lorenz Attractor")


def create_figure(plots, figsize=(12, 5), face_color = "#395979"):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(face_color)

    for i, (func, projection) in enumerate(plots, start=1):
        ax = fig.add_subplot(1, len(plots), i, projection=projection)
        func(ax)

    plt.tight_layout()
    return fig


def main():
    create_figure(
        [
            (generate_dem_plot, "3d"),
            (generate_polar_plot, "polar"),
        ]
    )

    create_figure(
        [
            (generate_torus_plot, "3d"),
            (generate_heatmap, None),
        ]
    )

    create_figure(
        [
            (generate_scatter3d, "3d"),
            (generate_trisurface_plot, "3d"),
        ]
    )

    create_figure(
        [
            (generate_helix_plot, "3d"),
            (generate_contour3d_plot, "3d"),
        ]
    )

    create_figure(
        [(generate_surface_plot, "3d")],
        figsize=(6, 5),
    )

    create_figure(
        [
            (generate_wireframe_plot, "3d"),
            (generate_bar3d_plot, "3d"),
        ]
    )

    create_figure(
        [
            (generate_quiver_plot, None),
            (generate_contourf_plot, None),
        ]
    )

    create_figure(
        [(generate_sphere_plot, "3d")],
        figsize=(6, 5),
    )

    create_figure(
        [(generate_lorenz_plot, "3d")],
        figsize=(6, 5),
    )

    plt.show()


if __name__ == "__main__":
    main()
