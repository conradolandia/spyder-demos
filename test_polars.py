#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 12:04:47 2025

@author: andi
"""

import polars as pl
from datetime import date

# Create a dictionary to hold data
data = {
    "name": ["Alice", "Bob", "Charlie", "David"],
    "age": [25, 30, 22, 35],
    "city": ["New York", "London", "Paris", "Tokyo"],
    "join_date": [
        date(2023, 1, 15),
        date(2022, 5, 20),
        date(2024, 3, 10),
        date(2023, 9, 1),
    ],
}

# Create a Polars DataFrame from the dictionary
df = pl.DataFrame(data)
