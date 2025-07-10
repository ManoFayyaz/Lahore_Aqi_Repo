import hopsworks
import pandas as pd
import numpy as np
import os

# Step 1: Login using API key
api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# Step 2: Load and clean CSV
df = pd.read_csv("lahore_aqi_features.csv")

# Convert timestamp to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Drop rows with invalid or missing timestamps
df.dropna(subset=["timestamp"], inplace=True)

# Clean main_pollutant
df['main_pollutant'] = df['main_pollutant'].astype(str).str.strip().replace("", np.nan)
df.dropna(subset=["main_pollutant"], inplace=True)

# Drop any rows with other nulls
#df.dropna(inplace=True)

# Sort by timestamp (optional but good practice)
df.sort_values("timestamp", inplace=True)

# Enforce data types explicitly for Hopsworks compatibility
df = df.astype({
    "aqius": "int64",
    "main_pollutant": "string",
    "temperature": "float64",
    "humidity": "float64",
    "wind_speed": "float64",
    "aqi": "float64",
    "pm2_5": "float64",
    "pm10": "float64",
    "no2": "float64",
    "so2": "float64",
    "co": "int64",
    "no": "float64",
    "o3": "float64",
    "nh3": "float64",
    "hour": "int64",
    "dayofweek": "int64",
    "target_aqi": "float64"
})

print("Shape of data before insert:", df.shape)


# Step 3: Create or get feature group
fg = fs.get_or_create_feature_group(
    name="lahore_aqi_group",
    version=1,
    primary_key=["timestamp"],
    description="Lahore AQI daily updated dataset",
    event_time="timestamp" 
)

# Step 4: Insert cleaned data
fg.insert(df, write_options={"wait_for_job": True})  # Don't overwrite

print("Feature group updated successfully!")
