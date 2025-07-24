from sklearn.model_selection import train_test_split
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
df= fg.select_all().read()

df.dropna(subset=["target_aqi"], inplace=True)
df.dropna(axis=1, how='all', inplace=True)

df["hour"]=df["timestamp"].dt.hour
df["dayofweek"]=df["timestamp"].dt.dayofweek

df['aqius_lag1'] = df['aqius'].shift(1)
df['aqius_lag2'] = df['aqius'].shift(2)
df['aqius_avg3'] = df['aqius'].rolling(window=3).mean()
df = df.dropna()  # important for lags/rolling

X=df[['so2', 'temperature', 'no', 'o3', 'humidity',"dayofweek"]]
y=df[['aqius']]

train=df.iloc[-144:-72]
test=df.iloc[-72:]

X_train=train[['so2', 'temperature', 'no', 'o3','wind_speed', 'humidity',"dayofweek","hour",'aqius_lag1', 'aqius_lag2', 'aqius_avg3']]
y_train=train[['aqius']]
X_test=test[['so2', 'temperature', 'no', 'o3', 'wind_speed','humidity',"dayofweek","hour",'aqius_lag1', 'aqius_lag2', 'aqius_avg3']]
y_test=test[['aqius']]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)

#  Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Model Training
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train.squeeze())
rf_preds = rf_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, rf_preds)
mae = mean_absolute_error(y_test, rf_preds)
r2 = r2_score(y_test, rf_preds)
rmse=np.sqrt(mse)

 # Save predictions & features
prediction_df = pd.DataFrame({
    'actual': y_test.squeeze(),
    'predicted': rf_preds
})

prediction_df.to_csv("prediction.csv", index=False)
X_test.to_csv("features.csv", index=False)

model_dir = "AQI_prediction_model"
os.makedirs(model_dir, exist_ok=True)

#  Save model
model_path = os.path.join(model_dir, "model.pkl")
joblib.dump(rf_model,model_path)

# Save predictions as CSV
prediction_path = os.path.join(model_dir, "predictions.csv")
prediction_df.to_csv(prediction_path, index=False)

#  Save input features for SHAP or review
features_path = os.path.join(model_dir, "features.csv")
X_test.to_csv(features_path, index=False)


#  Upload full folder to Model Registry
mr = project.get_model_registry()
model = mr.python.create_model(
    name="AQI_prediction_model",
    metrics={"mae":mae,"mse":mse,"rmse":rmse,"r2":r2},
    description=f"Best model for predicting AQI: Random Forest Regression"
)
model.save(os.path.abspath(model_dir))

print(" Model, predictions, features, and metrics uploaded to registry.")
