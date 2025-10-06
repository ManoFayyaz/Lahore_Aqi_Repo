import pandas as pd
from dash import Dash, html, dcc
import plotly.express as px

# --- Load Data ---
data = pd.read_csv("lahore_aqi_data.csv")

data = data.dropna(subset=['aqius'])
data = data.dropna(subset=['pm2_5'])
data = data.dropna(subset=['pm10'])
data = data.dropna(subset=['temperature'])
data = data.dropna(subset=['humidity'])



# If timestamp is string, convert to datetime

# --- Initialize App ---
app = Dash(__name__)

data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
data = data.sort_values("timestamp")

# --- Charts ---
# 1. AQI over time
fig_aqi = px.line(
    data, x="timestamp", y="aqius",
    title="Air Quality Index (AQI) Over Time",
    markers=True,
    color_discrete_sequence=["#2E86AB"]
)

fig_aqi.update_layout(
    template="plotly_dark",        # Dark theme
    plot_bgcolor="black",          # Plot area background
    paper_bgcolor="black",         # Around the plot
    font_color="white",            # Make text visible
    xaxis_title="Timestamp",
    yaxis_title="AQI",
)

# 2. PM2.5 vs PM10
fig_pm = px.line(
    data, x="timestamp", y=["o3", "pm10"],
    title="PM2.5 and PM10 Concentration Over Time",
    markers=True
)

fig_pm.update_layout(
    template="plotly_dark",        # Dark theme
    plot_bgcolor="black",          # Plot area background
    paper_bgcolor="black",         # Around the plot
    font_color="white",            # Make text visible
    xaxis_title="Timestamp",
    yaxis_title="AQI",
)

# 3. Temperature and Humidity
fig_weather = px.line(
    data, x="timestamp", y=["temperature", "humidity"],
    title="Temperature and Humidity Trends",
    markers=True
)

fig_weather.update_layout(
    template="plotly_dark",        # Dark theme
    plot_bgcolor="black",          # Plot area background
    paper_bgcolor="black",         # Around the plot
    font_color="white",            # Make text visible
    xaxis_title="Timestamp",
    yaxis_title="AQI",
)

app.layout = html.Div(
    style={
        "backgroundColor": "#000000",
        "color": "white",
        "padding": "20px",
        "minHeight": "100vh",
        "width": "90%",
        "margin": "auto"
    },
    children=[
        html.H1("🌆 Lahore Air Quality Dashboard", style={"textAlign": "center"}),

        html.Div([
            dcc.Graph(figure=fig_aqi, style={"marginBottom": "30px"}),
            dcc.Graph(figure=fig_pm, style={"marginBottom": "30px"}),
            dcc.Graph(figure=fig_weather)
        ])
    ]
)

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)

