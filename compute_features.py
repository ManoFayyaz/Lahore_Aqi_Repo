import pandas as pd

df = pd.read_csv("lahore_aqi_data.csv", parse_dates=["timestamp"])


df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

df['target_aqi'] = df['aqius'].shift(-1)

df = df.dropna()

df.to_csv("lahore_aqi_features.csv", index=False)

print("Features saved to lahore_aqi_features.csv!")
