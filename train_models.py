# from sklearn.model_selection import TimeSeriesSplit
# import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import hopsworks
import os

# Login to Hopsworks and load data
api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="lahore_aqi_group", version=1)
df = fg.select_all().read()

#sorting data that hopsworks shuffled up
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values("timestamp")
df = df.reset_index(drop=True)

# Preprocessing 
df.dropna(subset=["target_aqi"], inplace=True)
df.dropna(axis=1, how='all', inplace=True)

last_timestamp = df['timestamp'].max()

# print(last_timestamp)

# Feature Engineering
df['pm2_5_lag1'] = df['pm2_5'].shift(1)
df['pm2_5_lag2'] = df['pm2_5'].shift(2)
df['pm2_5_avg3'] = df['pm2_5'].rolling(window=3).mean()
df['pm2_5_avg7']  = df['pm2_5'].rolling(window=7).mean()   
df['pm2_5_std7']  = df['pm2_5'].rolling(window=7).std()   
df['pm2_5_avg14']  = df['pm2_5'].rolling(window=14).mean()   
df['pm2_5_std14']  = df['pm2_5'].rolling(window=14).std() 
df['pm2_5_max']=df['pm2_5'].rolling(window=7).max()  

df = df.dropna()

df['month'] = df['timestamp'].dt.month
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Last 72 rows = Test set (future data)
train = df.iloc[:-72, :]
test = df.iloc[-72:]

features = ['temperature','no2', 'co', 'no', 'o3','humidity','pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_avg3','pm2_5_avg7','pm2_5_std7','pm2_5_avg14','pm2_5_std14','pm2_5_max','dayofweek','month']
target = 'pm2_5'

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Final Model Training 
xgb_model =XGBRegressor(n_estimators=200,max_depth=4,learning_rate=0.1,reg_alpha=0.2,reg_lambda=2.0,subsample=0.6,colsample_bytree=0.6,random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)

# Evaluate Final Test Set
mse = mean_squared_error(y_test, xgb_preds)
mae = mean_absolute_error(y_test, xgb_preds)
r2 = r2_score(y_test, xgb_preds)
rmse=np.sqrt(mse)
mape = np.mean(np.abs((y_test - xgb_preds) / y_test)) * 100

# print("MSE :", mse)
# print("MAE :", mae)
# print("RMSE:", rmse)
# print("R2  :", r2)
# print("MAPE Test: {:.2f}%".format(mape_test))

#US EPA Formula: AQI
BP_high=None
BP_low=None
AQI_high=None
AQI_low=None
aqi=[]

for i in xgb_preds:
  if i>=0.0 and i<= 9.0:
      BP_low=0.0
      BP_high=9.0
      AQI_low=0
      AQI_high=50
  elif i>=9.1 and i<=35.4:
      BP_low=9.1
      BP_high=35.4
      AQI_low=51
      AQI_high=100
  elif i>=35.5 and i<=55.4:
      BP_low=35
      BP_high=55.4
      AQI_low=101
      AQI_high=150
  elif i>=55.5 and i<=125.4:
      BP_low=55.5
      BP_high=125.4
      AQI_low=151
      AQI_high=200
  elif i>=125.5 and i<=225.4:
      BP_low=125.5
      BP_high=225.4
      AQI_low=152
      AQI_high=300
  elif i>=225.5 and i<=325.4:
      BP_low=225.5
      BP_high=325.4
      AQI_low=301
      AQI_high=500
  AQI=((AQI_high-AQI_low)*(i-BP_low))/(BP_high-BP_low)+AQI_low
  aqi.append(AQI)

#creating future timestamps
future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1),periods=len(y_test),freq='H')

# Save predictions, calculations & features
prediction_df = pd.DataFrame({"timestamp": future_timestamps,"actual":y_test.values,'predicted_pm2_5':xgb_preds,"calculated_aqi":aqi})
prediction_df.to_csv("prediction.csv", index=False)
X_test.to_csv("features.csv", index=False)

model_dir = "AQI_prediction_model"
os.makedirs(model_dir, exist_ok=True)

# Saving model
model_path = os.path.join(model_dir, "model.pkl")
joblib.dump(xgb_model, model_path)

# Save predictions and test features
prediction_df.to_csv(os.path.join(model_dir, "predictions.csv"), index=False)
X_test.to_csv(os.path.join(model_dir, "features.csv"), index=False)

# Upload to Hopsworks Model Registry
mr = project.get_model_registry()
model = mr.python.create_model(
    name="AQI_prediction_model",
    metrics={"mae": mae, "mse": mse, "rmse": rmse, "r2": r2,"mape":mape},
    description="PM2.5 predictions and Calculated AQI"
)
model.save(os.path.abspath(model_dir))
print("Model, predictions, features, and metrics uploaded to registry.")
