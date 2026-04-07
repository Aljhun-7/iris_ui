import pandas as pd
import joblib
import gdown
from sklearn.tree import DecisionTreeClassifier

# Download and Load
file_id = "1FBCKZI4KKtlvZaY8XgvAcvLHo5We0ORF"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
gdown.download(url, "Iris.csv", quiet=False)
df = pd.read_csv("Iris.csv")

# Train
X = df.drop(columns=["Id","Species"])
y = df["Species"]
model = DecisionTreeClassifier()
model.fit(X, y)

# SAVE THE FILE
joblib.dump(model, 'iris_model.joblib')
print("✅ iris_model.joblib created successfully!")
