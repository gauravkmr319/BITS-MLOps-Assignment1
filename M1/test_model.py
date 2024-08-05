# test_model.py
import joblib
import numpy as np
from pathlib import Path
import pytest

path = Path(__file__).parent.parent
print(path)
model = joblib.load(f'{path}/test_res/model.pkl')


def test_model():
    assert model.predict(np.array([[4]])) == pytest.approx(4, rel=1e-2)


if __name__ == "__main__":
    test_model()
