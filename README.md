# Fraud-Detection-ML

Fraud Detection 

Detect fraudulent credit card transactions using Isolation Forest.


1 train_model.py
# Train fraud detection model
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Sample data
data = pd.DataFrame({"amount":[50,20,30,45,60,5000,40,35,25,7000]})

model = IsolationForest(contamination=0.2, random_state=42)
model.fit(data[["amount"]])

joblib.dump(model, "fraud_model.pkl")
print("Model is ready")

2 predict.py
# Test the model
import pandas as pd
import joblib

model = joblib.load("fraud_model.pkl")
new_data = pd.DataFrame({"amount":[40,8000,70,15000]})

new_data["prediction"] = model.predict(new_data[["amount"]])
print(new_data)

3 requirements.txt
pandas
scikit-learn
joblib


Detect fraudulent credit card transactions using Isolation Forest.

