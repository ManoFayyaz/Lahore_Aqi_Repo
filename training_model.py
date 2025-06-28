import hopsworks

project = hopsworks.login(
    api_key_value="secrets.HOPWORKS_API_KEY",  # Replace with your real API key or use environment variable
    project="Lahore_aqi"
)
fs = project.get_feature_store()

fg = fs.get_feature_group(name="lahore_aqi_group", version=1)
