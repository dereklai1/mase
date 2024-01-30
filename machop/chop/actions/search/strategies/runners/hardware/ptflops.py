import torch

from .base import HWRunnerBase
from chop.passes.graph.analysis.flop_estimator.ptflops import ptflops_module_analysis_pass


class RunnerPtFlops(HWRunnerBase):

    available_metrics = ("flops", )

    def _post_init_setup(self) -> None:
        pass

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        """
        Run the model on the data_loader for num_batches, and return a dict of metrics in the form of dict[str, float]
        """

        if not isinstance(model, torch.nn.Module):
            pass_model = model.model
        else:
            pass_model = model

        data_loader = data_module.val_dataloader()
        dummy_in, _ = next(iter(data_loader))

        _, metrics = ptflops_module_analysis_pass(pass_model, {
            "dummy_in": dummy_in[0]
        })

        return metrics
