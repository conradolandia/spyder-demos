import numpy as np
import pandas as pd
import polars as pl

# NUMERIC TYPES
# -------------


# Integer (int) - whole numbers
e = 42
print(f"Integer: {e}, Type: {type(e)}")

# Float - decimal numbers
f = 3.14159
print(f"Float: {f}, Type: {type(f)}")

# Complex - numbers with real and imaginary parts
c = 3 + 4j
print(f"Complex: {c}, Type: {type(c)}")


# SEQUENCE TYPES
# -------------

# String (str) - sequence of characters
s = "Hello, Python!"
print(f"String: {s}, Type: {type(s)}")

# List - ordered, mutable collection
l = [1, 2, 3, "four", 5.0]
print(f"List: {l}, Type: {type(l)}")

# Tuple - ordered, immutable collection
t = (1, 2, 3, "four", 5.0)
print(f"Tuple: {t}, Type: {type(t)}")

# Range - sequence of numbers
r = range(5)  # 0, 1, 2, 3, 4
print(f"Range: {r}, Type: {type(r)}")


# MAPPING TYPE
# ------------

# Dictionary (dict) - key-value pairs
d = {"name": "Python", "version": 3.10, "is_fun": True}
print(f"Dictionary: {d}, Type: {type(d)}")


# SET TYPES
# ---------

# Set - unordered collection of unique items
st = {1, 2, 3, 4, 5}
print(f"Set: {st}, Type: {type(st)}")

# Frozen Set - immutable set
fs = frozenset([1, 2, 3, 4, 5])
print(f"Frozen Set: {fs}, Type: {type(fs)}")


# BOOLEAN TYPE
# -----------

# Boolean (bool) - True or False
plots = True
equations = False
print(f"Boolean True: {plots}, Type: {type(plots)}")
print(f"Boolean False: {equations}, Type: {type(equations)}")


# BINARY TYPES
# -----------

# Bytes - immutable sequence of bytes
string_bytes = b"hello"
print(f"Bytes: {string_bytes}, Type: {type(string_bytes)}")

# Bytearray - mutable sequence of bytes
string_bytearray = bytearray(b"hello")
print(f"Bytearray: {string_bytearray}, Type: {type(string_bytearray)}")

# Memoryview - memory view of an object
mem_map = memoryview(string_bytes)
print(f"Memoryview: {mem_map}, Type: {type(mem_map)}")


# DATAFRAME TYPES
# ---------------

with np.load("jacksboro_fault_dem.npz") as dem:
    elevation = dem["elevation"]
    nrows, ncols = elevation.shape
    x_coords = np.linspace(dem["xmin"], dem["xmax"], ncols)
    y_coords = np.linspace(dem["ymin"], dem["ymax"], nrows)

# Subsample the DEM grid into a tabular sample
row_idx = np.linspace(0, nrows - 1, 50, dtype=int)
col_idx = np.linspace(0, ncols - 1, 50, dtype=int)
rr, cc = np.meshgrid(row_idx, col_idx, indexing="ij")

dem_table = {
    "x": x_coords[cc].ravel(),
    "y": y_coords[rr].ravel(),
    "elevation": elevation[rr, cc].ravel(),
}

df_pandas = pd.DataFrame(dem_table)
print(f"Pandas DataFrame: shape={df_pandas.shape}, Type: {type(df_pandas)}")

df_polars = pl.DataFrame(dem_table)
print(f"Polars DataFrame: shape={df_polars.shape}, Type: {type(df_polars)}")
