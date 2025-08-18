import gradio as gr 
import pandas as pd 
import matplotlib.pyplot as plt 
import hopsworks 
from datetime import datetime, timedelta, timezone 
import numpy as np 
#Connect to Hopsworks 
project = hopsworks.login( api_key_value="RI6aVh8JRlgiuVaz.bGgoZw1u0Lf54YkBoZyivKakNFHWMHcQE3z5hCk4GOpTbHKf7jHLol2cXmSfZSMC", project="Lahore_aqi" )
fs = project.get_feature_store() 
model_registry = project.get_model_registry()

# Load model + artifacts
models = model_registry.get_models(name="AQI_prediction_model") 
latest_model = sorted(models, key=lambda m: m.version)[-1] 
model_dir = latest_model.download() 

# Next 3 days AQI (from predictions.csv) 
def prediction_chart(): 
    predictions_path = model_dir + "/predictions.csv" 
    df = pd.read_csv(predictions_path) 
    if len(df) != 72: raise ValueError(f"Expected 72 hourly predictions, but got {len(df)}")
    # Split into 3 days and get max AQI for each day 
    daily_max = [df["predicted_pm2_5"][i*24:(i+1)*24].max() for i in range(3)]
    # Generate next 3 dates
    days = [(datetime.today() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(3)] 
    # Plot
    plt.figure(figsize=(10, 4)) 
    plt.barh(days, daily_max, color='blue')
    plt.title("Maximum PM2.5 for Next 3 Days") 
    plt.xlabel("Date") 
    plt.ylabel("Max PM2.5") 
    # plt.grid(True) 
    plt.tight_layout()
    return plt.gcf() 
    
def pm2_5_line_chart(): 
    predictions_path = model_dir + "/predictions.csv" 
    df = pd.read_csv(predictions_path) 
    start_time = datetime.now() + timedelta(hours=1)
    timestamps = [start_time + timedelta(hours=i) for i in range(len(df))] 
    df["timestamp"] = timestamps 
    fig, ax = plt.subplots(figsize=(12, 4)) 
    ax.plot(df["timestamp"], df['predicted_pm2_5'], marker='o', linestyle='-', color='blue') 
    ax.set_xlabel("Date") 
    ax.set_ylabel("Predicted PM2.5") 
    ax.set_title(" PM2.5 Forecast (Next 72 Hours)") 
    plt.tight_layout() 
    return fig 


def aqi_line_chart(): 
    predictions_path = model_dir + "/predictions.csv" 
    df = pd.read_csv(predictions_path) 
    start_time = datetime.now() + timedelta(hours=1)
    timestamps = [start_time + timedelta(hours=i) for i in range(len(df))] 
    df["timestamp"] = timestamps 
    fig, ax = plt.subplots(figsize=(12, 4)) 
    ax.plot(df["timestamp"], df['calculated_aqi'], marker='o', linestyle='-', color='blue') 
    ax.set_xlabel("Date") 
    ax.set_ylabel("Calculated AQI") 
    ax.set_title("AQI Forecast (Next 72 Hours)") 
    plt.tight_layout() 
    return fig 
    
def alert_dataframe(): 
    predictions_path = model_dir + "/predictions.csv" 
    df = pd.read_csv(predictions_path)
    daily_max = [round(df["calculated_aqi"][i*24:(i+1)*24].max(),1) for i in range(3)]
    days = [(datetime.today() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(3)] 
    
    alerts = [] 
    for date, value in zip(days, daily_max):
        if value >=0 and value<=50: 
            level ="Good" 
        elif value >=51 and value<=100: 
            level = "Moderate " 
        elif value >= 101 and value<=150: 
            level = "Unhealthy " 
        elif value >=151 and value<=300: 
            level = "Unhealthy" 
        elif value >=301 and value<=400: 
            level="Hazourdous" 
        elif value >=500: 
            level="Hazourdous" 
        alerts.append({"Date": date, "Maximum AQI": value, "AQI Category": level}) 
        
    return pd.DataFrame(alerts) 


# def prediction_line_chart():
#     predictions_path = model_dir + "/predictions.csv"
#     df = pd.read_csv(predictions_path) 
#     start_time = datetime.now() + timedelta(hours=1) 
#     timestamps = [start_time + timedelta(hours=i) for i in range(len(df))] 
#     df["timestamp"] = timestamps 
#     fig, ax = plt.subplots(figsize=(12, 4)) 
#     ax.plot(df["timestamp"], df["calculated_aqi"], marker='o', linestyle='-', color='blue') 
#     ax.set_xlabel("Date") 
#     ax.set_ylabel("Calculated AQI") 
#     ax.set_title("Next 72 Hours AQI") 
#     plt.xticks(rotation=45) 
#     plt.tight_layout() 
#     return fig 
    
# DASHBOARD ONLY 
with gr.Blocks() as dashboard: 
        gr.Markdown("\n<h1>PM2.5 Predictions of Lahore</h1>\n") 
        with gr.Row(): 
            with gr.Column():
                gr.Plot(value=pm2_5_line_chart(), label="PM2.5 Forecast (Next 72 Hours)") 
                
        gr.Markdown("<h2>Alerts!!!!</h2>") 
        with gr.Row(): 
            with gr.Column(): 
                gr.Dataframe(value=alert_dataframe(), headers=["Date", "Maximum_AQI", "Forecast"], interactive=False)
                
        gr.Markdown("\n<h1>Calculated AQI through US EPA</h1>\n")  
        with gr.Row(): 
            with gr.Column(): 
                gr.Plot(value=aqi_line_chart(),label="72 Hours Forecast of AQI")
            
        gr.Markdown("<h2>Maximum PM2.5 </h2>") 
        with gr.Row(): 
            with gr.Column(): 
                 gr.Plot(value= prediction_chart(), label="Max PM2.5") 
                    
# LAUNCH DASHBOARD 
dashboard.launch()
