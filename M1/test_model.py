# test_model.py
import joblib
import numpy as np
import pytest

import utils

path = utils.get_project_root()
model = joblib.load(f'{path}/model.pkl')

def test_model():
    assert model.predict(np.array([[4]])) == pytest.approx(4, rel=1e-2)

if __name__ == "__main__":
    test_model()
