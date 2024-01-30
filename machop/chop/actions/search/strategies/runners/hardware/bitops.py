from .base import HWRunnerBase
from chop.passes.graph.analysis.quantization.calculate_bitops import calculate_bitops_analysis_pass


class RunnerBitOPs(HWRunnerBase):

    available_metrics = ("bitops", )

    def _post_init_setup(self) -> None:
        pass

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        """
        Run the model on the data_loader for num_batches, and return a dict of metrics in the form of dict[str, float]
        """
        _, metrics = calculate_bitops_analysis_pass(model)

        return metrics
