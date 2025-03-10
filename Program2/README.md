# TOPSIS Method for Decision Analysis
This Python package provides an efficient implementation of the TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) method. It's a straightforward tool to evaluate alternatives based on multiple criteria and rank them according to their relative performance.

## How to Install
To install the package, use pip as follows:

pip install topsis102203635

## Quick Start Guide
Import the Library and Set Up Data
Here's an example to help you get started with this package:

import pandas as pd
from topsis102203635 import Topsis


data = {
    'P1': [0.88, 0.91, 0.79, 0.94],
    'P2': [0.77, 0.83, 0.62, 0.88],
    'P3': [6.5, 7.0, 4.8, 3.6],
    'P4': [51.5, 31.7, 46.7, 62.2],
    'P5': [14.91, 10.11, 13.23, 16.91]
}

df = pd.DataFrame(data)

weights = [0.2, 0.2, 0.2, 0.2, 0.2]
impacts = ['+', '+', '-', '+', '+']

topsis = Topsis(df, weights, impacts)

result = topsis.calculate()

print(result)

# Key Features
Ease of Use: Simplified syntax for implementing the TOPSIS method.
Flexible Criteria: Support for any number of criteria with user-defined weights and impacts.
Customizable: Easily modify input data, weights, and the direction (impact) of each criterion.