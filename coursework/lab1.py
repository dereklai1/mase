import os
import itertools
from datetime import datetime
from os import path
import json


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

def mkdir_if_not_exist(dir: str):
    os.makedirs(dir, exist_ok=True)

def train(cfg: dict, output_dir: str, id: int = 0):
    train_project_name = f"jsc-train-{id}"
    mkdir_if_not_exist(path.join(output_dir, train_project_name))
    cmd = [
        "./ch",
        "train", # Action
        "jsc-tiny", # Model
        "jsc", # Dataset
        "--accelerator", "cpu", # CPU is faster for tiny models
        "--project-dir", output_dir,
        "--project", train_project_name,
        "--max-epochs", str(cfg["epochs"]),
        "--batch-size", str(cfg["batch-size"]),
        "--learning-rate", str(cfg["learning-rate"]),
        "|", "tee", path.join(output_dir, train_project_name, "train.log")
    ]
    with open(path.join(output_dir, train_project_name, "config.json"), 'w') as f:
        json.dump(cfg, f, indent=4)
    os.system(" ".join(cmd))

def test(output_dir: str, id: int = 0):
    test_project_name = f"jsc-test-{id}"
    mkdir_if_not_exist(path.join(output_dir, test_project_name))
    test_log_file = path.join(output_dir, test_project_name, "test.log")
    results_txt = path.join(output_dir, test_project_name, "result.txt")
    results_json = path.join(output_dir, test_project_name, "result.json")
    cmd = [
        "./ch",
        "test",
        "jsc-tiny", # Model
        "jsc", # Dataset
        "--project-dir", output_dir,
        "--project", test_project_name,
        "--load", f"{output_dir}/jsc-train-{id}/software/training_ckpts/best.ckpt",
        "--load-type", "pl", # Pytorch lightning ckpt
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


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    config = {
        "epochs": [5, 10, 20, 40],
        "batch-size": [64, 256, 1024],
        "learning-rate": [0.00001, 0.001, 0.1]
    }
    output_dir = f"../lab1_output/{timestamp}"
    for id, cfg in enumerate(product_dict(**config)):
        train(cfg, output_dir, id)
        test(output_dir, id)
