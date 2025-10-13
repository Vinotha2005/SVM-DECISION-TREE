# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import numpy as np
import pickle

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

# Drop unnecessary columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode categorical variables
le_geo = LabelEncoder()
le_gender = LabelEncoder()

df["Geography"] = le_geo.fit_transform(df["Geography"])
df["Gender"] = le_gender.fit_transform(df["Gender"])

# Features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = ridge_model.predict(X_test)
y_pred_class = np.where(y_pred >= 0.5, 1, 0)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Accuracy:", accuracy_score(y_test, y_pred_class))

# Save model and scaler
with open("churn_model.pkl", "wb") as f:
    pickle.dump(ridge_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully (churn_model.pkl, scaler.pkl)")
