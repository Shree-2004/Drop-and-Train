import numpy as np

def flatten_to_1d(X):
    return np.asarray(X).ravel()

import joblib

model = joblib.load("best_model_rf_20251123_114127.joblib")
print("Model loaded!")

# show entire model
print(model)
