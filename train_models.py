import hopsworks
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import joblib

api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

fg = fs.get_feature_group(name="lahore_aqi_group", version=1)
df = fg.select_all().read()

print(df.columns)
print(df.head(10))
print(df.info())
print(df.describe())

print(df.isnull().sum())
df.dropna(inplace=True)


# Select features and target
features = ['temperature', 'humidity', 'wind_speed', 'aqius']
target = 'target_aqi'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Ridge & Deep Learning models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


#RidgerRegression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_preds = ridge_model.predict(X_test_scaled)


# Step 1: Define features and target
features = ['temperature', 'humidity', 'wind_speed', 'aqius']
target = 'target_aqi'

X = df[features]
y = df[target]

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Scale features (StandardScaler for X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Scale target (MinMaxScaler for y)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# Step 5: Build TensorFlow Model
tf_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])

# Step 6: Compile model
optimizer = Adam(learning_rate=0.001)
tf_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Step 7: Train the model
tf_model.fit(X_train_scaled, y_train_scaled, epochs=200, batch_size=32, verbose=0)

# Step 8: Make predictions
tf_preds_scaled = tf_model.predict(X_test_scaled)
tf_preds = y_scaler.inverse_transform(tf_preds_scaled).flatten()

# Step 9: Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, tf_preds))
mae = mean_absolute_error(y_test, tf_preds)
r2 = r2_score(y_test, tf_preds)

# Step 10: Print results
print("\n📊 TensorFlow Model (Improved) Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.2f}")


#Evaluating the best one
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


rf_rmse, rf_mae, rf_r2 = evaluate_model(y_test, rf_preds)
ridge_rmse, ridge_mae, ridge_r2 = evaluate_model(y_test, ridge_preds)
tf_rmse, tf_mae, tf_r2 = evaluate_model(y_test,tf_preds)

metrics = {
    "RandomForest": {"r2": rf_r2, "model": rf_model},
    "Ridge": {"r2": ridge_r2, "model": ridge_model},
    "TensorFlow": {"r2": tf_r2, "model": tf_model}
}

# Pick the model with the highest R²
best_model_name = max(metrics, key=lambda name: metrics[name]["r2"])
best_model = metrics[best_model_name]["model"]

print(f"✅ Best model based on R²: {best_model_name} (R² = {metrics[best_model_name]['r2']:.2f})")


# Log in to Hopsworks
project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project="Lahore_aqi"
)
mr = project.get_model_registry()

# 📁 Save best model to local directory
model_dir = "best_model"
os.makedirs(model_dir, exist_ok=True)

if best_model_name == "TensorFlow":
    model_path = os.path.join(model_dir, "model.keras")
    best_model.save(model_path)
else:
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(best_model, model_path)

# 📊 Recalculate metrics for best model
if best_model_name == "RandomForest":
    preds = rf_model.predict(X_test)
elif best_model_name == "Ridge":
    preds = ridge_model.predict(X_test_scaled)
else:
    preds = tf_model.predict(X_test_scaled).flatten()

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

# 🗂️ Register model as `"ridge_aqi_model"` (name is fixed)
model = mr.python.create_model(
    name="ridge_aqi_model",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description=f"Best model auto-selected: {best_model_name}"
)

# 📤 Upload to model registry
model.save(os.path.abspath(model_path))

print(f"✅ Model '{best_model_name}' registered as 'ridge_aqi_model' with R² = {r2:.2f}")
