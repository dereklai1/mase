import logging

import pandas as pd
from tabulate import tabulate

from ...utils import get_mase_op, get_mase_type, get_node_actual_target


logger = logging.getLogger(__name__)


def travese_graph_analysis_pass(graph) -> None:
    """
    Traverses a graph and prints out some info collected.
    """

    columns = [
        "Torch Name",
        # "Torch Op",
        # "Torch Args",
        # "Torch Kwargs",
        "MASE Type",
        "Mase Op",
        "Common Params",
        "Software Params",
        "Hardware Params",
    ]
    rows = list()

    for n in graph.fx_graph.nodes:
        rows.append([
            n.name,
            # n.op,
            # n.args,
            # n.kwargs,
            get_mase_type(n),
            get_mase_op(n),
            n.meta['mase'].parameters['common'],
            n.meta['mase'].parameters['software'],
            n.meta['mase'].parameters['hardware'],
        ])

    logger.info("\n" + tabulate(rows, headers=columns, tablefmt="orgtbl"))

    return pd.DataFrame(rows, columns=columns)
