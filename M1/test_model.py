# test_model.py
import joblib
import numpy as np
import pytest

model = joblib.load('model.pkl')

def test_model():
    assert model.predict(np.array([[4]])) == pytest.approx(4, rel=1e-2)

if __name__ == "__main__":
    test_model()
