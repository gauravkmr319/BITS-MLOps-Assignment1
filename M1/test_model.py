# test_model.py
from pathlib import Path
import joblib
import numpy as np
import pytest

path = Path(__file__).parent.parent
print(path)
model = joblib.load(f'{path}/test_res/model.pkl')


def test_model():
    assert model.predict(np.array([[4]])) == pytest.approx(4, rel=1e-2)


if __name__ == "__main__":
    test_model()
