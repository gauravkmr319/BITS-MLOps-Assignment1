# model.py
import os
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Sample Data
X = np.array([[1], [2], [3]])
y = np.array([1, 2, 3])

# Model
model = LinearRegression()
model.fit(X, y)

# Save Model
path = Path(os.path.dirname(os.path.realpath(__file__)))
print(f'{path}/test_res/model.pkl')
joblib.dump(model, f'{path}\\model.pkl')
