import pandas as pd

# Safely load the CSV and skip broken rows if any
df = pd.read_csv("lahore_aqi_data.csv", on_bad_lines='skip', parse_dates=["timestamp"])

# Extract hour and day of week from timestamp
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Target: AQI of the next time step (e.g. next hour or next record)
df['target_aqi'] = df['aqius'].shift(-1)

# Drop rows with any missing values
df = df.dropna()

# Save to new CSV
df.to_csv("lahore_aqi_features.csv", index=False)

print("✅ Features saved to lahore_aqi_features.csv!")
