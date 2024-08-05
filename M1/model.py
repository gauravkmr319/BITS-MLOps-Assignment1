# model.py
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import joblib

# Sample Data
X = np.array([[1], [2], [3]])
y = np.array([1, 2, 3])

# Model
model = LinearRegression()
model.fit(X, y)

# Save Model
path = Path(__file__).parent.parent
joblib.dump(model, f'{path}/test_res/model.pkl')
