#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h2o
import ydata_profiling
import gradio as gr
from h2o.automl import H2OAutoML
from ydata_profiling import ProfileReport

# Initialize H2O
h2o.init()

def read_csv(file):
    return pd.read_csv(file)

def generate_profile_report(data):
    profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
    profile.to_file("profile_report.html")
    return "profile_report.html"

def train_automl_model(data, target_column):
    h2o_data = h2o.H2OFrame(data)
    train, test = h2o_data.split_frame(ratios=[.8])
    x = train.columns
    x.remove(target_column)
    y = target_column

    aml = H2OAutoML(max_runtime_secs=600, seed=1)
    aml.train(x=x, y=y, training_frame=train)

    leaderboard = aml.leaderboard.as_data_frame()
    model_id = aml.leader.model_id
    return model_id, leaderboard

def analyze_and_train(file, target_column):
    data = read_csv(file)
    profile_path = generate_profile_report(data)
    model_id, leaderboard = train_automl_model(data, target_column)
    return profile_path, model_id, leaderboard.to_html()

interface = gr.Interface(
    fn=analyze_and_train,
    inputs=[gr.File(label="Upload CSV"), gr.Textbox(label="Target Column for Prediction")],
    outputs=[
        gr.File(label="Profile Report (HTML)"),
        gr.Textbox(label="Best Model ID"),
        gr.HTML(label="Leaderboard")
    ],
    title="Data Smart: AutoML and Profiling for Tabular Data",
    description="Upload a CSV file to generate a profile report and train an AutoML model."
)

if __name__ == "__main__":
    interface.launch()