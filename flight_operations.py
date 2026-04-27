#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon May  4 17:07:07 2020

@author: juanis
"""

# pylint: disable=invalid-name

# Imports
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.spatial import KDTree, Voronoi, voronoi_plot_2d

plt.style.use("dark_background")


# File paths
file_paths = ["airports_CO.dat", "borders_CO.dat"]

airports_CO = file_paths[0]
borders_CO = file_paths[1]


# Read files (fallback dummy data if files don't exist)
try:
    airports_Col = pd.read_csv(
        airports_CO,
        sep=r"\s+",
        names=[
            "coord-y",
            "coord-x",
            "Altitude",
            "City",
            "Department",
            "Airport",
        ],
    )
except Exception:
    # Dummy data
    airports_Col = pd.DataFrame(
        {
            "coord-x": np.random.uniform(-80, -70, 20),
            "coord-y": np.random.uniform(-5, 10, 20),
            "Altitude": np.random.randint(0, 3000, 20),
            "City": ["City"] * 20,
            "Department": ["Dept"] * 20,
            "Airport": [f"A{i}" for i in range(20)],
        }
    )

try:
    borders_Col = pd.read_csv(
        borders_CO, sep=r"\s+", names=["coord-y", "coord-x"]
    )
except Exception:
    # Dummy polygon-like border
    theta = np.linspace(0, 2 * np.pi, 100)
    borders_Col = pd.DataFrame(
        {"coord-x": -75 + 5 * np.cos(theta), "coord-y": 3 + 5 * np.sin(theta)}
    )


# Main Class
class FlightOperations:

    def __init__(self, airports, borders):
        self.airports = airports
        self.borders = borders

        self.points = self.airports[["coord-x", "coord-y"]].to_numpy()

        # Spatial structures
        self.tree = KDTree(self.points)
        self.vor = Voronoi(self.points)

    def sleep_wrapper(self):
        time.sleep(0.003)

    def plotAirports(self):
        """Plot map with airports"""
        fig, ax = plt.subplots()

        voronoi_plot_2d(
            self.vor, ax=ax, show_vertices=False, line_colors="orange"
        )

        ax.plot(
            self.borders["coord-x"],
            self.borders["coord-y"],
            color="white",
            linewidth=1,
        )

        ax.scatter(
            self.airports["coord-x"],
            self.airports["coord-y"],
            color="cyan",
            s=10,
        )

        ax.set_title("Airports + Voronoi Regions")
        plt.show()

    def findNearestPointKD(self, point):
        """Find nearest airport given a point using KDTree"""
        distance, idx = self.tree.query(point)
        return {
            "index": idx,
            "distance": distance,
            "airport": self.airports.iloc[idx],
        }


# Example usage
if __name__ == "__main__":
    ops = FlightOperations(airports_Col, borders_Col)

    # Plot
    ops.plotAirports()

    # Query example
    test_point = np.array([-74.0, 4.5])
    result = ops.findNearestPointKD(test_point)

    print("Query point:", test_point)
    print("Nearest airport index:", result["index"])
    print("Distance:", result["distance"])
    print("Airport row:\n", result["airport"])
