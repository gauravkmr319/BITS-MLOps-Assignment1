# model.py
# import os
# from pathlib import Path
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
PATH = '/home/runner/work/BITS-MLOps-Assignment1/BITS-MLOps-Assignment1'
print(f'{PATH}/test_res/model.pkl')
joblib.dump(model, f'{PATH}/test_res/model.pkl')
