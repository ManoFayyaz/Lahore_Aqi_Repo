import requests
import csv
from datetime import datetime
import os

# Your API key and URL
api_key = "c80b5ffb-d106-4bac-8b41-2d2d660f0d91"
url = f"https://api.airvisual.com/v2/city?city=Lahore&state=Punjab&country=Pakistan&key={api_key}"

# Send request to API
response = requests.get(url)
data = response.json()

# Get the useful data
aqi = data['data']['current']['pollution']['aqius']
main_pollutant = data['data']['current']['pollution']['mainus']
temperature = data['data']['current']['weather']['tp']
humidity = data['data']['current']['weather']['hu']
wind = data['data']['current']['weather']['ws']

time_now = datetime.now().isoformat()

file_path = 'lahore_aqi_data.csv'
file_empty = not os.path.exists(file_path) or os.path.getsize(file_path) == 0

# Save to CSV file
with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file)

    if file_empty:
        writer.writerow(['timestamp', 'aqius', 'main_pollutant', 'temperature', 'humidity', 'wind_speed'])

    writer.writerow([time_now, aqi, main_pollutant, temperature, humidity, wind])

print("Data saved.")
