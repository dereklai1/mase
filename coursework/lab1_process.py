import pandas as pd
import os
from os import path
from glob import glob
import re
from pathlib import Path
import json
import altair as alt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def json_to_dict(file: str):
    with open(file, 'r') as f:
        j = json.load(f)
    return j

def process_tests(dir: str):
    data = {
        "epochs": [],
        "batch_size": [],
        "learning_rate": [],
        "test_acc": [],
        "test_loss": [],
        "test_acc_steps": [] # list of dataframes
    }
    training_folders = glob(path.join(dir, "jsc-train*"))
    for train_folder in training_folders:

        event_log_folder = path.join(
            train_folder, "software/tensorboard/lightning_logs/version_0"
        )
        event_log = glob(path.join(event_log_folder, "events*"))[0]
        assert path.exists(event_log)
        ea = EventAccumulator(event_log)
        ea.Reload()
        train_acc_steps = ea.Scalars('train_acc_step')

        train_acc_data = []
        # Normalise integer steps between 0-1 so easier to graph
        max_len_steps = len(train_acc_steps)
        for step, scalar in enumerate(train_acc_steps):
            train_acc_data.append((step / max_len_steps, scalar.value))
        data["test_acc_steps"].append(train_acc_data)

        # Process Summary Dataframe
        test_folder = train_folder.replace('train', 'test')
        assert path.exists(test_folder)
        config = json_to_dict(path.join(train_folder, "config.json"))
        results = json_to_dict(path.join(test_folder, "result.json"))
        data["epochs"].append(config["epochs"])
        data["batch_size"].append(config["batch-size"])
        data["learning_rate"].append(config["learning-rate"])
        data["test_acc"].append(results["test_acc_epoch"])
        data["test_loss"].append(results["test_loss_epoch"])

    return pd.DataFrame(data)

def plot_batchsize_summary(df: pd.DataFrame):
    data = df[df['epochs'] == 10]
    base = alt.Chart(data).mark_line().properties(
        width=400,
        height=200
    )
    base.encode(
        x='batch_size:N',
        y='test_acc:Q',
        color='learning_rate:N'
    ).save("batch_acc.png")
    base.encode(
        x='batch_size:N',
        y='test_loss:Q',
        color='learning_rate:N'
    ).save("batch_loss.png")

def plot_batchsize_steps(df: pd.DataFrame):
    data = df[(df['epochs'] == 10) & (df["learning_rate"] == 0.00100)]
    data = data.drop(['epochs', 'learning_rate', 'test_loss', 'test_acc'], axis=1)
    data = data.explode('test_acc_steps')
    split_steps = pd.DataFrame(data['test_acc_steps'].to_list(), index=data.index)
    data[['normalised_step', 'accuracy']] = split_steps
    data = data.drop(['test_acc_steps'], axis=1)
    alt.Chart(data).mark_line(
        strokeWidth=1,
        interpolate='basis',
        opacity=0.9
    ).encode(
        x='normalised_step',
        y=alt.Y('accuracy').scale(zero=False),
        color='batch_size:N'
    ).properties(
        width=800,
        height=400
    ).save("batchsize_training_steps.png")

def plot_epoch(df: pd.DataFrame):
    data = df[df['batch_size'] == 256]
    alt.Chart(data).mark_line().encode(
        x='epochs:N',
        y=alt.Y('test_acc').scale(zero=False),
        color='learning_rate:N'
    ).properties(
        width=800,
        height=400
    ).save("epoch_test_acc.png")

def plot_learning_rate_batch(df: pd.DataFrame):
    data = df[df['epochs'] == 20]
    alt.Chart(data).mark_rect().encode(
        x='learning_rate:O',
        y='batch_size:O',
        color='test_acc'
    ).properties(
        width=600,
        height=600
    ).save("learning_rate_vs_batch.png")


if __name__ == '__main__':
    run_dir = "../lab1_output/24-01-23-22-53-01"
    df = process_tests(run_dir)
    plot_batchsize_steps(df)
    plot_epoch(df)
    plot_learning_rate_batch(df)
