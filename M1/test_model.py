# test_model.py
from pathlib import Path
import os
import joblib
import numpy as np
import pytest

path = Path(os.path.dirname(os.path.realpath(__file__)))
print("here")
print(path)
model = joblib.load(f'{path}\\model.pkl')


def test_model():
    a = 5
    b = 4
    assert a-b == 1


if __name__ == "__main__":
    test_model()
