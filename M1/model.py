# model.py
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([[1], [2], [3]])
y = np.array([1, 2, 3])

# Model
model = LinearRegression()
model.fit(X, y)

# Save Model
import joblib
joblib.dump(model, 'model.pkl')
