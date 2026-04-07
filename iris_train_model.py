import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 1. Load some sample data
data = load_iris()
X, y = data.data, data.target

# 2. Train the model
model = RandomForestClassifier()
model.fit(X, y)

# 3. Save the model to a file
# This creates 'iris_model.joblib' in your current folder
joblib.dump(model, 'iris_model.joblib')

print("Model saved successfully!")
