#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse import linalg as spla
from scipy import interpolate
from scipy import fft
from scipy import optimize
from typing import Dict


# %% GRID GENERATION


def generate_grid(n: int = 200, length: float = 1.0):
    """
    Generate a 1D uniform grid.
    """

    x = np.linspace(0, length, n)
    dx = x[1] - x[0]

    return x, dx


# %% LAPLACIAN OPERATOR


def build_laplacian(n: int, dx: float):
    """
    Construct the finite-difference Laplacian operator.
    """

    main = -2 * np.ones(n)
    off = np.ones(n - 1)
    lap = sparse.diags([off, main, off], [-1, 0, 1])

    return lap / dx**2


# %% LINEAR SYSTEM SOLVER


def solve_poisson(n: int = 200):
    """
    Solve a 1D Poisson equation using sparse linear algebra.
    """

    x, dx = generate_grid(n)
    lap = build_laplacian(n, dx)
    f = np.sin(np.pi * x)
    u = spla.spsolve(lap, f)

    return x, u


# %% HEAT EQUATION


def heat_equation(n: int = 150, steps: int = 200):
    """
    Simulate a heat equation using explicit Euler time stepping.
    """

    x, dx = generate_grid(n)
    lap = build_laplacian(n, dx)
    dt = 0.4 * dx**2
    u = np.exp(-100 * (x - 0.5) ** 2)
    history = [u.copy()]

    for _ in range(steps):
        u = u + dt * (lap @ u)
        history.append(u.copy())

    return x, np.array(history)


# %% EIGENVALUE PROBLEM


def laplacian_eigenmodes(n: int = 200):
    """
    Compute eigenmodes of the Laplacian operator.
    """

    x, dx = generate_grid(n)
    lap = build_laplacian(n, dx)
    vals, vecs = spla.eigs(lap, k=6, which="SM")

    return x, vals.real, vecs.real


# %% INTERPOLATION EXAMPLE


def interpolation_demo():
    """
    Demonstrate cubic interpolation.
    """

    x = np.linspace(0, 10, 20)
    y = np.sin(x)

    spline = interpolate.CubicSpline(x, y)
    x_dense = np.linspace(0, 10, 400)

    return x, y, x_dense, spline(x_dense)


# %% FFT EXAMPLE


def fft_demo(n: int = 4096):
    """
    Compute FFT of a synthetic signal.
    """

    t = np.linspace(0, 1, n)

    signal = np.sin(2 * np.pi * 40 * t) + 0.7 * np.sin(2 * np.pi * 90 * t)

    spectrum = fft.fft(signal)
    freq = fft.fftfreq(n, d=t[1] - t[0])

    return freq, np.abs(spectrum)


# %% MONTE CARLO INTEGRATION


def monte_carlo_integral(n_samples: int = 300000):
    """
    Estimate an integral using Monte Carlo sampling.
    """

    rng = np.random.default_rng()
    x = rng.uniform(0, 1, n_samples)
    values = np.exp(-(x**2))

    return values.mean()


# %% OPTIMIZATION EXAMPLE


def optimize_function():
    """
    Minimize the Rosenbrock function.
    """

    def rosen(x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    x0 = np.zeros(4)
    res = optimize.minimize(rosen, x0)

    return res.x


# %% PLOTTING


def create_plots(results):
    """
    Generate example scientific plots.
    """

    x, poisson = results["poisson"]
    heat_x, heat_history = results["heat"]
    freq, spectrum = results["fft"]
    ix, iy, ix_dense, iy_dense = results["interp"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Poisson solution
    axes[0, 0].plot(x, poisson)
    axes[0, 0].set_title("Poisson equation solution")

    # Heat equation final state
    axes[0, 1].plot(heat_x, heat_history[0], label="initial")
    axes[0, 1].plot(heat_x, heat_history[-1], label="final")
    axes[0, 1].set_title("Heat equation evolution")
    axes[0, 1].legend()

    # FFT spectrum
    axes[1, 0].plot(freq, spectrum)
    axes[1, 0].set_xlim(0, 150)
    axes[1, 0].set_title("FFT spectrum")

    # Interpolation
    axes[1, 1].scatter(ix, iy)
    axes[1, 1].plot(ix_dense, iy_dense)
    axes[1, 1].set_title("Cubic spline interpolation")

    plt.tight_layout()

    return fig


# %% PIPELINE


def run_example() -> Dict:

    poisson = solve_poisson()
    heat = heat_equation()
    eigen = laplacian_eigenmodes()
    interp = interpolation_demo()
    fft_res = fft_demo()
    mc = monte_carlo_integral()
    opt = optimize_function()

    results = {
        "poisson": poisson,
        "heat": heat,
        "eigen": eigen,
        "interp": interp,
        "fft": fft_res,
        "montecarlo": mc,
        "optimum": opt,
    }

    print(results)
    return results


# %% MAIN

if __name__ == "__main__":

    results = run_example()
    fig = create_plots(results)
    plt.show()
