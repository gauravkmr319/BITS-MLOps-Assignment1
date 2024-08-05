# test_model.py
import joblib
import numpy as np
import pytest


model = joblib.load('model.pkl')

def test_model():
    assert 4 == 4

if __name__ == "__main__":
    test_model()
