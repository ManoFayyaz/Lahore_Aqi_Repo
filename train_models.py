import hopsworks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import joblib


api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
fs=project.get_feature_store()

fg = fs.get_feature_group(name="lahore_aqi_group", version=1)
df= fg.select_all().read()
print(df.columns)
print(df.head())

df.dropna(subset=["target_aqi"], inplace=True)
df.dropna(axis=1, how='all', inplace=True)

X=df[['so2', 'temperature', 'no', 'o3', 'humidity']]
y=df[['aqius']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_preds = ridge_model.predict(X_test_scaled)

#RFG
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


ridge_metrics=evaluate(y_test, ridge_preds)
rf_metrics=evaluate(y_test, rf_preds)

#best model
models = {
    "Ridge": (ridge_model, ridge_metrics, True),
    "RandomForest": (rf_model, rf_metrics, False)
}

for name, val in models.items():
    print(f"{name}: {val}")
valid_models = {k: v for k,
                v in models.items()
                 if v is not None and v[1] is not None and 'r2' in v[1]}

# Now safely get the best model
best_model_name = max(valid_models, key=lambda k: valid_models[k][1]['r2'])
best_model, best_metrics, is_scaled = valid_models[best_model_name]

print(f"Best model: {best_model_name}")
print(f"Metrics: {best_metrics}")


# ✅ Save predictions as a CSV inside model_dir
predictions = best_model.predict(X_test_scaled if is_scaled else X_test).flatten()
prediction_df = pd.DataFrame(
    np.column_stack((y_test.values, predictions)),
    columns=["y_true", "y_pred"]
)
model_dir = "best_model"
os.makedirs(model_dir, exist_ok=True)

# Save model
model_path = os.path.join(model_dir, "model.pkl")
joblib.dump(best_model, model_path)

# ✅ Save predictions to same folder
prediction_path = os.path.join(model_dir, "predictions.csv")
prediction_df.to_csv(prediction_path, index=False)

# Upload model directory with both model and predictions
mr = project.get_model_registry()
model = mr.python.create_model(
    name="AQI_prediction_model",
    metrics=best_metrics,
    description=f"Best model for predicting AQI: {best_model_name}"
)
model.save(os.path.abspath(model_dir))  # ✅ Save entire directory

print("✅ Model and predictions uploaded to registry.")
