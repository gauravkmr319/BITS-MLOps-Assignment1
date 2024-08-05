# ml_experiment.py
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import numpy as np


def train_and_log_model(learning_rate, n_samples=100):
    # Generate synthetic data
    X, y = make_regression(n_samples=n_samples, n_features=1, noise=0.1)

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict and calculate metrics
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    # Log experiment with MLflow
    with mlflow.start_run():
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
        print(f"Logged run with MSE: {mse}")


# Train and log multiple models with different parameters
learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    train_and_log_model(lr)
