import requests
import csv
from datetime import datetime
import os

# === API CONFIGURATION ===
IQAIR_API_KEY = "c80b5ffb-d106-4bac-8b41-2d2d660f0d91"
OPENWEATHER_API_KEY = "d795e69b955f35e7baecbeee690c6630"
LAT, LON = "31.5497", "74.3436"

# === GET DATA FROM IQAir ===
iqair_url = f"https://api.airvisual.com/v2/city?city=Lahore&state=Punjab&country=Pakistan&key={IQAIR_API_KEY}"
iqair_response = requests.get(iqair_url).json()

aqius = iqair_response['data']['current']['pollution']['aqius']
main_pollutant = iqair_response['data']['current']['pollution']['mainus']
temperature = iqair_response['data']['current']['weather']['tp']
humidity = iqair_response['data']['current']['weather']['hu']
wind_speed = iqair_response['data']['current']['weather']['ws']

# === GET POLLUTANTS FROM OPENWEATHER ===
ow_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
ow_response = requests.get(ow_url).json()
components = ow_response['list'][0]['components']

pm2_5 = components.get("pm2_5")
pm10 = components.get("pm10")
no2 = components.get("no2")
so2 = components.get("so2")
co = components.get("co")
no = components.get("no")
o3 = components.get("o3")
nh3 = components.get("nh3")

# === PREPARE FINAL RECORD ===
timestamp = datetime.now().isoformat()

record = [
    timestamp, aqius, main_pollutant, temperature, humidity, wind_speed,
    pm2_5, pm10, no2, so2, co, no, o3, nh3
]

# === SAVE TO CSV ===
csv_path = "lahore_aqi_data.csv"
file_empty = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)

    if file_empty:
        writer.writerow([
            "timestamp", "aqius", "main_pollutant", "temperature", "humidity", "wind_speed",
            "pm2_5", "pm10", "no2", "so2", "co", "no", "o3", "nh3"
        ])
    
    writer.writerow(record)

print("✅ Combined AQI and pollutant data saved.")
