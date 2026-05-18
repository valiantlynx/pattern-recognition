"""Run Task 1: EDA only."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import FashionMNISTData
from main import task1_eda

data = FashionMNISTData()
task1_eda(data)
print("\nTask 1 complete!")
