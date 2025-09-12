import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

data = pd.DataFrame({
    "amount": [50,20,30,45,60,5000,40,35,25,7000]
})

model = IsolationForest(contamination=0.2, random_state=42)
model.fit(data[["amount"]])

data["prediction"]=model.predict(data[["amount"]])

print(data)