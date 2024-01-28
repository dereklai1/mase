import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "machop"
    )
)

from chop.passes.graph import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    save_mase_graph_interface_pass,
)
from chop.dataset import MaseDataModule
from chop.tools.checkpoint_load import load_model
from chop.models import get_model_info, get_model
from chop.ir.graph.mase_graph import MaseGraph
from chop.tools.get_input import InputGenerator
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.logger import set_logging_verbosity
from chop.passes.graph.utils import get_mase_op, get_mase_type


if __name__ == "__main__":

    set_logging_verbosity("info")

    # Config
    CHECKPOINT_PATH = "/home/derek/mase/coursework/lab1/best-custom-nn/best.ckpt"
    MODEL_NAME = "jsc-m"
    DATASET_NAME = "jsc"
    QUANTISE_INFO = {
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
        },
    }

    # Get data module
    data_module = MaseDataModule(
        name=DATASET_NAME,
        batch_size=8,
        model_name=MODEL_NAME,
        num_workers=0,
    )
    data_module.prepare_data()
    data_module.setup()

    # Get jsc-m model and load checkpoint
    model = get_model(
        MODEL_NAME,
        task="cls",
        dataset_info=data_module.dataset_info,
        pretrained=False
    )
    model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)

    # Get input generator
    model_info = get_model_info(MODEL_NAME)
    input_generator = InputGenerator(
        data_module=data_module,
        model_info=model_info,
        task="cls",
        which_dataloader="train",
    )
    dummy_in = next(iter(input_generator))

    # Apply initial metadata passes
    mg = MaseGraph(model=model)
    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in , "add_value": True})

    # Apply quantization pass and compare
    # ori_mg = deepcopy_mase_graph(mg)
    mg, _ = quantize_transform_pass(mg, QUANTISE_INFO)
    # summarize_quantization_analysis_pass(ori_mg, mg)

    # Check that they are actually quantized by applying quantisation transform
    # fixed ()
    for n in mg.fx_graph.nodes:
        if get_mase_op(n) != "linear":
            continue
        print(n.name, get_mase_type(n), get_mase_op(n))
        data_in = n.meta['mase'].parameters['common']['args']['data_in_0']
        weight = n.meta['mase'].parameters['common']['args']['weight']
        bias = n.meta['mase'].parameters['common']['args']['bias']

        weight_tensor = weight["value"]
        weight_tensor = weight_tensor * (2 ** 4)
        print(data_in)
        print(weight_tensor)
        print(bias)
        break
