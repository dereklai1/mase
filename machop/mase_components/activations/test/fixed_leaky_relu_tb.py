import logging

import cocotb

from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.testbench import Testbench

from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t


class FixedLeakyReLUTB(Testbench):

    def __init__(self, dut, clk=None, rst=None) -> None:
        super().__init__(dut, clk, rst)

        self.assign_self_params([
            "DATA_IN_0_PRECISION_0", "DATA_IN_0_PRECISION_1",
            "DATA_IN_0_TENSOR_SIZE_DIM_0", "DATA_IN_0_TENSOR_SIZE_DIM_1",
            "DATA_IN_0_PARALLELISM_DIM_0", "DATA_IN_0_PARALLELISM_DIM_1",
            "DATA_OUT_0_PRECISION_0", "DATA_OUT_0_PRECISION_1",
            "DATA_OUT_0_TENSOR_SIZE_DIM_0", "DATA_OUT_0_TENSOR_SIZE_DIM_1",
            "DATA_OUT_0_PARALLELISM_DIM_0", "DATA_OUT_0_PARALLELISM_DIM_1",
            "SLOPE_TYPE", "POWER_OF_2_SLOPE",
        ])

        self.data_in_driver = StreamDriver(
            clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.data_out_monitor = StreamMonitor(
            clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
        )

    def generate_inputs(self, batches=1):
        return super().generate_inputs(batches)

@cocotb.test()
async def basic(dut):
    tb = FixedLeakyReLUTB(dut, in_features=20, out_features=20)


if __name__ == "__main__":

    DEFAULT_CONFIG = {
        "DATA_IN_0_PRECISION_0": 8,
        "DATA_IN_0_PRECISION_1": 1,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": 4,
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 2,
        "DATA_OUT_0_PRECISION_0": 8,
        "DATA_OUT_0_PRECISION_1": 1,
        "DATA_OUT_0_TENSOR_SIZE_DIM_0": 4,
        "DATA_OUT_0_TENSOR_SIZE_DIM_1": 4,
        "DATA_OUT_0_PARALLELISM_DIM_0": 2,
        "DATA_OUT_0_PARALLELISM_DIM_1": 2,

        "SLOPE_TYPE": "power-of-2",
        "POWER_OF_2_SLOPE": 2,
    }

    mase_runner(
        module_param_list=[
            DEFAULT_CONFIG,
        ],
        trace=True
    )
