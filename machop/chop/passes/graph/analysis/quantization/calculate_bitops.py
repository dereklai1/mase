import logging

import numpy as np


logger = logging.getLogger(__name__)


def calculate_bitops_analysis_pass(graph, pass_args: dict = {}):
    """
    Calculate the number of BitOPs there are in the graph.
    """

    total_bitop_cost = 0

    for node in graph.fx_graph.nodes:
        mase_meta = node.meta["mase"].parameters
        mase_op = mase_meta["common"]["mase_op"]
        mase_type = mase_meta["common"]["mase_type"]

        if mase_type in ["module", "module_related_func"]:

            if mase_op == "linear":

                data_in_0_meta = mase_meta["common"]["args"]["data_in_0"]
                w_meta = mase_meta["common"]["args"]["weight"]

                input_width = data_in_0_meta["precision"][0]
                weight_width = w_meta["precision"][0]
                weight_shape = w_meta["shape"]

                # BitOP cost for a single multiplcation
                mult_bitop_cost = input_width * weight_width
                num_mults = np.prod(weight_shape)
                layer_bitop_cost = mult_bitop_cost * num_mults

                logger.debug(f"Input: width={input_width}")
                logger.debug(f"Weight: width={weight_width}, shape={weight_shape}")

            else:
                logger.debug(f"Not implemented for {mase_op}")
                layer_bitop_cost = 0

            total_bitop_cost += layer_bitop_cost

    return graph, {"bitops": total_bitop_cost}
