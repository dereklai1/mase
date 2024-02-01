import os
import itertools
from datetime import datetime
from os import path
import json
import numpy as np
import pandas as pd
import os
from os import path
from glob import glob
import re
from pathlib import Path
import json
import altair as alt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from random import randint


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

def mkdir_if_not_exist(dir: str):
    os.makedirs(dir, exist_ok=True)

def train(
        cfg: dict,
        output_dir: str,
        id: int = 0,
        model: str = "mnist_lab4_relu",
        dataset: str = "mnist",
        accel: str = "gpu",
        extra_args: list[str] = []
    ):
    train_project_name = f"train-{id}"
    mkdir_if_not_exist(path.join(output_dir, train_project_name))
    # train_log_file = path.join(output_dir, train_project_name, "train.log")
    seed = randint(0, 2**31)
    cmd = [
        "./ch",
        "train", # Action
        model,
        dataset,
        "--accelerator", accel,
        "--project-dir", output_dir,
        "--project", train_project_name,
        "--max-epochs", str(cfg["epochs"]),
        "--batch-size", str(cfg["batch-size"]),
        "--learning-rate", str(cfg["learning-rate"]),
        "--weight-decay", str(cfg["weight-decay"]),
        "--seed", str(seed),
        *extra_args,
        # "|", "tee", train_log_file
    ]
    with open(path.join(output_dir, train_project_name, "config.json"), 'w') as f:
        json.dump(cfg | {"seed": seed}, f, indent=4)
    os.system(" ".join(cmd))
    # os.system(f"rm {train_log_file}")

def test(
        output_dir: str,
        id: int = 0,
        model: str = "mnist_lab4_relu",
        dataset: str = "mnist",
        extra_args: list[str] = []
    ):
    test_project_name = f"test-{id}"
    mkdir_if_not_exist(path.join(output_dir, test_project_name))
    test_log_file = path.join(output_dir, test_project_name, "test.log")
    results_txt = path.join(output_dir, test_project_name, "result.txt")
    results_json = path.join(output_dir, test_project_name, "result.json")
    cmd = [
        "./ch",
        "test",
        model,
        dataset,
        "--project-dir", output_dir,
        "--project", test_project_name,
        "--load", f"{output_dir}/train-{id}/software/training_ckpts/best.ckpt",
        "--load-type", "pl", # Pytorch lightning ckpt
        *extra_args,
        "|", "tee", test_log_file
    ]
    os.system(" ".join(cmd))
    os.system("tail -n 3 " + test_log_file + " | awk '$1 ~ /test.+/ {print $1, $2;}' > " + results_txt)
    result_dict = dict()
    with open(results_txt, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            k, v = line.split()
            result_dict[k] = float(v)
    with open(results_json, 'w') as f:
        json.dump(result_dict, f, indent=4)
    os.system(f"rm {test_log_file}")

def json_to_dict(file: str):
    with open(file, 'r') as f:
        j = json.load(f)
    return j

def process_tests(dir: str):
    data = {
        "dir": [],
        "epochs": [],
        "batch_size": [],
        "learning_rate": [],
        "best_train_acc": [],
        "test_acc": [],
        "test_loss": [],
        # "weight_decay": [],
        "test_acc_steps": [], # list of dataframes
    }
    training_folders = glob(path.join(dir, "train*"))
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
        data["dir"].append(train_folder)
        data["best_train_acc"].append(best_train_acc)
        data["test_acc_steps"].append(train_acc_data)

        # Process Summary Dataframe
        test_folder = train_folder.replace('train', 'test')
        assert path.exists(test_folder), f"{test_folder} does not exist!"
        config = json_to_dict(path.join(train_folder, "config.json"))
        results = json_to_dict(path.join(test_folder, "result.json"))
        data["epochs"].append(config["epochs"])
        data["batch_size"].append(config["batch-size"])
        data["learning_rate"].append(config["learning-rate"])
        # data["weight_decay"].append(config["weight-decay"])
        data["test_acc"].append(results["test_acc_epoch"])
        data["test_loss"].append(results["test_loss_epoch"])

    return pd.DataFrame(data)

def plot_batchsize_steps(output_dir: str, data: pd.DataFrame):
    data = data.explode('test_acc_steps')
    split_steps = pd.DataFrame(data['test_acc_steps'].to_list(), index=data.index)
    data[['normalised_step', 'train_accuracy']] = split_steps
    data = data.drop(['test_acc_steps', 'test_loss', 'test_acc'], axis=1)
    alt.Chart(data).mark_line(
        strokeWidth=1,
        interpolate='basis',
        opacity=0.9
    ).encode(
        x='normalised_step',
        y=alt.Y('train_accuracy').scale(zero=False),
        color='model:N',
        # detail='weight_decay:N',
    ).properties(
        width=800,
        height=400
    ).save(path.join(output_dir, "validation_acc.png"))

def lab4_train(timestamp: str, model_name: str):
    config = {
        "epochs": [20],
        "batch-size": [128],
        "learning-rate": [0.0001],
        "weight-decay": [1e-7],
    }
    output_dir = f"/home/derek/mase/lab4_output/{model_name}-{timestamp}"
    for id, cfg in enumerate(product_dict(**config)):
        train(cfg, output_dir, id=id, model=model_name)
        test(output_dir, id=id, model=model_name)
    data = process_tests(output_dir)
    data["model"] = model_name
    plot_batchsize_steps(output_dir, data)
    return data

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    data1 = lab4_train(timestamp, "mnist_lab4_relu")
    data2 = lab4_train(timestamp, "mnist_lab4_leakyrelu")
    data = pd.concat([data1, data2], axis=0)
    print(data)
    plot_batchsize_steps("/home/derek/mase/coursework/lab4/img_temp", data)
