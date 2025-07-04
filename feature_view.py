import hopsworks
import os

# Login
api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# Get the feature group
fg = fs.get_feature_group("lahore_aqi_group", version=1)

# Create or update feature view
fv = fs.get_or_create_feature_view(
    name="lahore_aqi_view",
    version=1,
    description="Feature view for AQI prediction using pollutants and time features",
    labels=["target_aqi"],  # prediction target
    query=fg.select([
        "timestamp",
        "temperature",
        "humidity",
        "wind_speed",
        "main_pollutant",
        "aqius",
        "pm2_5",
        "pm10",
        "no2",
        "so2",
        "co",
        "no",
        "o3",
        "nh3",
        "hour",
        "dayofweek",
        "target_aqi"
    ])
)

print(" Feature view updated or created.")
