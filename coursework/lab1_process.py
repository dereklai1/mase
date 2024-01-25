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
        "best_train_acc": [],
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
        best_train_acc = 0
        for step, scalar in enumerate(train_acc_steps):
            train_acc_data.append((step / max_len_steps, scalar.value))
            best_train_acc = max(best_train_acc, scalar.value)
        data["best_train_acc"].append(best_train_acc)
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

def plot_epoch():
    run_dir1 = "../lab1_output/sweep-24-01-23-22-53-01"
    df1 = process_tests(run_dir1)
    df1 = df1[df1['batch_size'] == 256]
    run_dir2 = "../lab1_output/long-epoch-24-01-24-20-32-44"
    df2 = process_tests(run_dir2)
    data = pd.concat([df1, df2], axis=0)
    base = alt.Chart(data)
    test = base.encode(
        x='epochs:N',
        y=alt.Y('test_acc').scale(zero=False),
        color='learning_rate:N'
    ).mark_line()
    train = base.encode(
        x='epochs:N',
        y=alt.Y('best_train_acc').scale(zero=False),
        color='learning_rate:N'
    ).mark_line(
        strokeDash=[8, 8]
    )

    (test + train).properties(
        width=800,
        height=400
    ).save("epoch_test_acc.png")

def plot_learning_rate_batch():
    run_dir = "../lab1_output/batch-learn-sweep-24-01-24-20-42-07"
    df = process_tests(run_dir)
    alt.Chart(df).mark_rect().encode(
        x='learning_rate:O',
        y='batch_size:O',
        color='test_acc'
    ).properties(
        width=600,
        height=600
    ).save("learning_rate_vs_batch.png")


if __name__ == '__main__':
    # plot_learning_rate_batch()
    plot_epoch()
