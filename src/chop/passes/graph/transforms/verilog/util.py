from chop.passes.graph.utils import vf
from chop.passes.graph.analysis.add_metadata.hardware_metadata_layers import (
    INTERNAL_COMP,
)


def get_verilog_parameters(graph):
    parameter_map = {}

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        node_name = vf(node.name)

        for key, value in (
            node.meta["mase"].parameters["hardware"]["verilog_param"].items()
        ):
            if not isinstance(value, (int, float, complex, bool)):
                value = '"' + value + '"'
            assert (
                f"{node_name}_{key}" not in parameter_map.keys()
            ), f"{node_name}_{key} already exists in the parameter map"
            parameter_map[f"{node_name}_{key}"] = value

    return parameter_map


def include_ip_to_project(node):
    """
    Copy internal files to the project
    """
    mase_op = node.meta["mase"].parameters["common"]["mase_op"]
    assert (
        mase_op in INTERNAL_COMP
    ), f"Cannot find mase op {mase_op} in internal components"
    return INTERNAL_COMP[mase_op][0]["dependence_files"]
