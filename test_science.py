#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 13:18:30 2025

@author: andi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate scientific data
data = {
    "Sample_ID": [f"PLNT_{i:03d}" for i in range(1, 51)],
    "Species": np.random.choice(
        [
            "Arabidopsis thaliana",
            "Zea mays",
            "Oryza sativa",
            "Solanum lycopersicum",
            "Triticum aestivum",
        ],
        50,
    ),
    "Genotype": np.random.choice(
        ["Wild Type", "Mutant A", "Mutant B", "Transgenic"], 50
    ),
    "Temperature_C": np.round(np.random.uniform(18, 32, 50), 1),
    "Humidity_%": np.random.randint(40, 95, 50),
    "Light_Intensity": np.random.choice(
        ["Low", "Medium", "High"], 50, p=[0.2, 0.5, 0.3]
    ),
    "Soil_pH": np.round(np.random.uniform(5.0, 8.5, 50), 2),
    "Leaf_Area_cm2": np.round(np.random.uniform(2.5, 35.0, 50), 2),
    "Chlorophyll_ug_cm2": np.round(np.random.uniform(10.5, 45.0, 50), 1),
    "Stomatal_Density": np.random.randint(80, 350, 50),
    "Flowering_Days": np.random.randint(14, 60, 50),
    "Biomass_g": np.round(np.random.uniform(0.15, 8.2, 50), 3),
    "Gene_Expression": np.round(np.random.uniform(0.01, 15.75, 50), 3),
}

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame info and first 5 rows
print("DataFrame Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 1. Basic statistics
print("\nDescriptive Statistics:")
print(df.describe())

# 2. Group by species and light intensity
print("\nMean Biomass by Species and Light:")
print(df.groupby(["Species", "Light_Intensity"])["Biomass_g"].mean().unstack())

# 3. Correlation matrix
print("\nCorrelation Matrix:")
print(df.select_dtypes(include="number").corr())

# 4. Filter by genotype
mutants = df[df["Genotype"].str.contains("Mutant")]
print(f"\nMutant Samples: {len(mutants)}")

# 5. Plotting examples
# 5.1 Boxplot
df.boxplot(column="Chlorophyll_ug_cm2", by="Species", figsize=(10, 6))
plt.title("Chlorophyll Content by Species")
plt.suptitle("")
plt.ylabel("Chlorophyll (μg/cm²)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5.2 Scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
for species in df["Species"].unique():
    species_data = df[df["Species"] == species]
    ax.scatter(
        species_data["Biomass_g"],
        species_data["Chlorophyll_ug_cm2"],
        label=species,
        alpha=0.6,
        s=50,
    )
ax.set_xlabel("Biomass (g)")
ax.set_ylabel("Chlorophyll (μg/cm²)")
ax.set_title("Biomass vs Chlorophyll Content by Species")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# 5.3 Histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df["Biomass_g"], bins=15, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Biomass (g)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Biomass")
axes[0].grid(True, alpha=0.3)

axes[1].hist(
    df["Chlorophyll_ug_cm2"],
    bins=15,
    edgecolor="black",
    alpha=0.7,
    color="green",
)
axes[1].set_xlabel("Chlorophyll (μg/cm²)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Chlorophyll Content")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5.4 Bar chart
fig, ax = plt.subplots(figsize=(10, 6))
mean_biomass = (
    df.groupby("Genotype")["Biomass_g"].mean().sort_values(ascending=False)
)
bars = ax.bar(
    mean_biomass.index, mean_biomass.values, alpha=0.7, edgecolor="black"
)
ax.set_xlabel("Genotype")
ax.set_ylabel("Mean Biomass (g)")
ax.set_title("Mean Biomass by Genotype")
ax.grid(True, alpha=0.3, axis="y")
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5.5 Heatmap (correlation matrix)
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = df.select_dtypes(include="number").columns
corr_matrix = df[numeric_cols].corr()
im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
ax.set_yticklabels(corr_matrix.columns)
ax.set_title("Correlation Matrix Heatmap")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Correlation Coefficient")
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(
            j,
            i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )
plt.tight_layout()
plt.show()

# 5.6 Violin plot
fig, ax = plt.subplots(figsize=(10, 6))
genotypes = df["Genotype"].unique()
data_by_genotype = [
    df[df["Genotype"] == g]["Gene_Expression"].values for g in genotypes
]
parts = ax.violinplot(
    data_by_genotype,
    positions=range(len(genotypes)),
    showmeans=True,
    showmedians=True,
)
ax.set_xticks(range(len(genotypes)))
ax.set_xticklabels(genotypes, rotation=45)
ax.set_ylabel("Gene Expression")
ax.set_title("Gene Expression Distribution by Genotype")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# 5.7 Line plot (grouped by category)
fig, ax = plt.subplots(figsize=(10, 6))
for light in df["Light_Intensity"].unique():
    light_data = df[df["Light_Intensity"] == light]
    mean_by_species = (
        light_data.groupby("Species")["Biomass_g"].mean().sort_values()
    )
    ax.plot(
        range(len(mean_by_species)),
        mean_by_species.values,
        marker="o",
        label=light,
        linewidth=2,
        markersize=8,
    )
ax.set_xticks(range(len(mean_by_species)))
ax.set_xticklabels(mean_by_species.index, rotation=45, ha="right")
ax.set_xlabel("Species")
ax.set_ylabel("Mean Biomass (g)")
ax.set_title("Mean Biomass by Species and Light Intensity")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
