import hopsworks
import pandas as pd
import os

# Step 1: Login using API key
api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# Step 2: Read full daily CSV
df = pd.read_csv("lahore_aqi_features.csv")
#Drop rows with any null values
df = df.dropna()

# Step 3: Create or get feature group
fg = fs.get_or_create_feature_group(
    name="lahore_aqi_group",
    version=1,
    primary_key=["timestamp"],
    description="Lahore AQI daily updated dataset"
    
)

# Step 4: Overwrite data
fg.insert(df, write_options={"wait_for_job": True, "overwrite": True})
