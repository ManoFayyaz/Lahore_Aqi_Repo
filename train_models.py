from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import hopsworks
import joblib
import os

api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

fg = fs.get_feature_group(name="lahore_aqi_group", version=1)
df = fg.select_all().read()

# Preprocessing and Feature Engineering
df.dropna(subset=["target_aqi"], inplace=True)
df.dropna(axis=1, how='all', inplace=True)

df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df['aqius_lag1'] = df['aqius'].shift(1)
df['aqius_lag2'] = df['aqius'].shift(2)
df['aqius_avg3'] = df['aqius'].rolling(window=3).mean()
df = df.dropna()

# Last 72 rows = Test set (future data)
train = df.iloc[:-72, :]
test = df.iloc[-72:]

features = ['so2', 'o3', 'temperature', 'co', 'no', 'wind_speed', 'aqius_lag1', 'aqius_lag2', 'aqius_avg3']
target = 'aqius'

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TimeSeriesSplit Cross-Validation on Training Data
tscv = TimeSeriesSplit(n_splits=5)
fold = 1
r2_scores = []

for train_index, val_index in tscv.split(X_train_scaled):
    X_fold_train, X_fold_val = X_train_scaled[train_index], X_train_scaled[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    fold_model = RandomForestRegressor(n_estimators=100, random_state=42)
    fold_model.fit(X_fold_train, y_fold_train)
    fold_preds = fold_model.predict(X_fold_val)

    r2 = r2_score(y_fold_val, fold_preds)
    r2_scores.append(r2)
    print(f"Fold {fold} R2 Score: {r2:.4f}")
    fold += 1

print(f"\nAverage CV R2 Score: {np.mean(r2_scores):.4f}")

# Final Model Training on Full Training Set
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

# Evaluate Final Test Set
mse = mean_squared_error(y_test, rf_preds)
mae = mean_absolute_error(y_test, rf_preds)
r2 = r2_score(y_test, rf_preds)
rmse = np.sqrt(mse)

print("\nFinal Test Set Evaluation:")
print("MSE :", mse)
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# Save predictions & features
prediction_df = pd.DataFrame({'actual': y_test.values, 'predicted': rf_preds})
prediction_df.to_csv("prediction.csv", index=False)
X_test.to_csv("features.csv", index=False)

model_dir = "AQI_prediction_model"
os.makedirs(model_dir, exist_ok=True)

# Save model
model_path = os.path.join(model_dir, "model.pkl")
joblib.dump(rf_model, model_path)

# Save predictions and test features
prediction_df.to_csv(os.path.join(model_dir, "predictions.csv"), index=False)
X_test.to_csv(os.path.join(model_dir, "features.csv"), index=False)

# Upload to Hopsworks Model Registry
mr = project.get_model_registry()
model = mr.python.create_model(
    name="AQI_prediction_model",
    metrics={"mae": mae, "mse": mse, "rmse": rmse, "r2": r2},
    description="Random Forest model with TimeSeriesSplit validation"
)
model.save(os.path.abspath(model_dir))

print("\n Model, predictions, features, and metrics uploaded to registry.")
print({"mae": mae, "mse": mse, "rmse": rmse, "r2": r2})
