import os, sys

sys.path.append("/home/derek/mase/machop")
sys.path.append("/home/derek/mase/")

from chop.ir.graph.mase_graph import MaseGraph

from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
)

from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    emit_cocotb_transform_pass,
    quantize_transform_pass,
)

from chop.tools.logger import set_logging_verbosity

import toml
import torch
import torch.nn as nn
from torch.nn import Module
from chop.actions import simulate

# from chop.models.vision. import MLP


# --------------- Define MASE Transform Pipeline ---------------

def hardware_emit_pipeline(model: Module):

    mg = MaseGraph(model=model)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 2, 2))
    dummy_in = {"x": x}

    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )

    config_file = os.path.join(
        os.path.abspath(""),
        "..",
        "..",
        "machop",
        "configs",
        "tests",
        "quantize",
        "fixed.toml",
    )
    with open(config_file, "r") as f:
        quan_args = toml.load(f)["passes"]["quantize"]
    mg, _ = quantize_transform_pass(mg, quan_args)

    _ = report_node_type_analysis_pass(mg)

    # Update the metadata
    for node in mg.fx_graph.nodes:
        for arg, arg_info in node.meta["mase"]["common"]["args"].items():
            if isinstance(arg_info, dict):
                arg_info["type"] = "fixed"
                arg_info["precision"] = [8, 3]
        for result, result_info in node.meta["mase"]["common"]["results"].items():
            if isinstance(result_info, dict):
                result_info["type"] = "fixed"
                result_info["precision"] = [8, 3]

    mg, _ = add_hardware_metadata_analysis_pass(mg, None)
    mg, _ = emit_verilog_top_transform_pass(mg)
    mg, _ = emit_internal_rtl_transform_pass(mg)
    mg, _ = emit_bram_transform_pass(mg)
    mg, _ = emit_cocotb_transform_pass(mg)

    simulate(skip_build=False, skip_test=False)


# --------------- Running Pipeline for both ---------------

MODELS = [
    ("relu", MLP(nn.ReLU, {})),
    ("leakyrelu", MLP(nn.LeakyReLU, {"negative_slope": 0.12})),
]

if __name__ == "__main__":

    set_logging_verbosity("debug")

    for name, model in MODELS:
        print(f"Running {name}...")
        hardware_emit_pipeline(model)
