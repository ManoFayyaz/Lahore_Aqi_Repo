import hopsworks
import os

api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# Get the feature group
fg = fs.get_feature_group("lahore_aqi_group", version=1)

# Create the feature view (only once)
fv = fs.get_or_create_feature_view(
    name="lahore_aqi_view",
    version=1,
    description="Feature view for AQI prediction",
    labels=["aqius"],  # Target column
    query=fg.select([
        "timestamp", "temperature", "humidity", "wind_speed", "main_pollutant", "aqius"
    ])
)

print("✅ Feature view created.")
