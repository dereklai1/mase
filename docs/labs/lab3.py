import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

sys.path.append(str(Path.cwd().parent.parent.joinpath("machop")))

# # figure out the correct path
# machop_path = Path(".").resolve().parent.parent /"machop"
# assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
# sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model


set_logging_verbosity("info")

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint = None)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

pass_args = {
"by": "type",
"default": {"config": {"name": None}},
"linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
},}

import copy
# build a search space
data_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
w_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
search_spaces = []
for d_config in data_in_frac_widths:
    for w_config in w_in_frac_widths:
        pass_args['linear']['config']['data_in_width'] = d_config[0]
        pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
        pass_args['linear']['config']['weight_width'] = w_config[0]
        pass_args['linear']['config']['weight_frac_width'] = w_config[1]
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_args))

# grid search
import torch
from torchmetrics.classification import MulticlassAccuracy

from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5
# This first loop is basically our search strategy,
# in this case, it is a simple brute force search

recorded_accs = []
for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    j = 0

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds = mg.model(xs)
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batchs:
            break
        j += 1
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    recorded_accs.append(acc_avg)


print(recorded_accs)
