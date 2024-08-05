# model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from M1 import utils

# Sample Data
X = np.array([[1], [2], [3]])
y = np.array([1, 2, 3])

# Model
model = LinearRegression()
model.fit(X, y)

# Save Model
path = utils.get_project_root()
joblib.dump(model, f'{path}/model.pkl')
