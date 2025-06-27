import feature_store as fs
import hopsworks
import os

api_key = os.getenv("RI6aVh8JRlgiuVaz.bGgoZw1u0Lf54YkBoZyivKakNFHWMHcQE3z5hCk4GOpTbHKf7jHLol2cXmSfZSMC")
project = hopsworks.login(api_key_value=api_key)
feature_view = fs.get_feature_view(name="lahore_aqi_data_view", version=1)

job = feature_view.create_training_data(
    description="Training dataset for Lahore AQI",
    data_format="csv",
    write_options={"wait_for_job": True},
    statistics_config={"enabled": True}

)

training_dataset, _ = feature_view.get_training_data(training_dataset_version=1)
