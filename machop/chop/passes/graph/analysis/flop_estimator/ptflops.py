import torch
import logging
from ptflops import get_model_complexity_info
from ptflops.pytorch_engine import get_flops_pytorch


logger = logging.getLogger(__name__)


def ptflops_module_analysis_pass(
    module: torch.nn.Module,
    pass_args: dict = {}
):

    assert isinstance(module, torch.nn.Module), "module must be a nn.Module instance"
    assert isinstance(pass_args, dict), "pass_args must be a dict instance"

    dummy_in = pass_args["dummy_in"]
    total_flops = 0

    for n, m in module.named_modules():

        if not (
            isinstance(m, torch.nn.Linear) or
            isinstance(m, torch.nn.BatchNorm1d) or
            isinstance(m, torch.nn.ReLU)
        ):
            continue

        with torch.cuda.device(0):
            flops, params = get_flops_pytorch(
                model=m,
                input_res=dummy_in.shape,
                print_per_layer_stat=False,
            )
            total_flops += flops
            logger.debug(f"{n}: FLOPS {flops}, Params {params}")

    return module, {
        "flops": total_flops,
    }
