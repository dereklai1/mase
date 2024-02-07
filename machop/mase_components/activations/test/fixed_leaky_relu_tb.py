import logging

import cocotb
from cocotb.triggers import Timer

import torch
from torch.nn.functional import leaky_relu

from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.testbench import Testbench
from mase_cocotb.utils import sign_extend, sign_extend_t, signed_to_unsigned
from mase_cocotb.runner import mase_runner
from mase_cocotb.matrix_utils import (
    gen_random_matrix_input,
    rebuild_matrix,
    split_matrix
)

logger = logging.getLogger("testbench")
logger.setLevel(logging.DEBUG)


class FixedLeakyReLUTB(Testbench):

    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        self.assign_self_params([
            "DATA_IN_0_PRECISION_0", "DATA_IN_0_PRECISION_1",
            "DATA_IN_0_TENSOR_SIZE_DIM_0", "DATA_IN_0_TENSOR_SIZE_DIM_1",
            "DATA_IN_0_PARALLELISM_DIM_0", "DATA_IN_0_PARALLELISM_DIM_1",
            "DATA_OUT_0_PRECISION_0", "DATA_OUT_0_PRECISION_1",
            "DATA_OUT_0_TENSOR_SIZE_DIM_0", "DATA_OUT_0_TENSOR_SIZE_DIM_1",
            "DATA_OUT_0_PARALLELISM_DIM_0", "DATA_OUT_0_PARALLELISM_DIM_1",
            "SLOPE", "POWER_OF_2_SLOPE", "FIXED_SLOPE",
        ])

        self.data_in_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )
        self.data_out_monitor = StreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
        )

        # Define some commonly used tuples
        self.tensor_dims = (
            self.DATA_IN_0_TENSOR_SIZE_DIM_0, self.DATA_IN_0_TENSOR_SIZE_DIM_1,
            self.DATA_IN_0_PARALLELISM_DIM_0, self.DATA_IN_0_PARALLELISM_DIM_1
        )

        self.precision = (
            self.DATA_IN_0_PRECISION_0, self.DATA_IN_0_PRECISION_1
        )

    def generate_inputs(self):
        return gen_random_matrix_input(*self.tensor_dims, *self.precision)

    def model(self, X_in):
        X = rebuild_matrix(X_in, *self.tensor_dims)

        if self.SLOPE == 1:  # "power-of-2"
            slope = 2 ** -self.POWER_OF_2_SLOPE
        elif self.SLOPE == 2:  # "fixed"
            signed_int_slope = sign_extend(
                self.FIXED_SLOPE, self.DATA_IN_0_PRECISION_0
            )
            slope = signed_int_slope / (2 ** self.DATA_IN_0_PRECISION_0)
        else:
            raise Exception("SLOPE doesn't exist.")

        # Change X from unsigned fixed into the signed floating point number
        # which it actually represents
        X = sign_extend_t(X, self.DATA_IN_0_PRECISION_0)
        X = X.to(torch.float32)
        X = X / (2 ** self.DATA_IN_0_PRECISION_1)

        # Call into torch.nn.F.leaky_relu for our model
        Y = leaky_relu(X, negative_slope=slope)

        logger.info("Leaky Relu Output")
        logger.info(Y)

        # Output Rounding and Clamping
        Y = torch.floor(Y * (2 ** self.DATA_OUT_0_PRECISION_1)).int()

        logger.info("Rounded")
        logger.info(Y)

        min_val = -(2**(self.DATA_OUT_0_PRECISION_0-1))
        max_val = (2**(self.DATA_OUT_0_PRECISION_0-1))-1
        Y = torch.clamp(Y, min_val, max_val)

        # Change back into unsigned representation for monitor output
        Y = signed_to_unsigned(Y, self.DATA_OUT_0_PRECISION_0)

        return split_matrix(Y, *self.tensor_dims)


@cocotb.test()
async def repeated_leaky_relu(dut):
    tb = FixedLeakyReLUTB(dut)
    await tb.reset()
    tb.data_out_monitor.ready.value = 1
    for _ in range(100):
        X = tb.generate_inputs()
        tb.data_in_driver.load_driver(X)
        exp_Y = tb.model(X)
        tb.data_out_monitor.load_monitor(exp_Y)
    await Timer(100, units="us")
    assert tb.data_out_monitor.exp_queue.empty()


def get_closest_fixed_slope(
    float_num: float,
    width: int
) -> int:
    assert 0.0 < float_num and float_num < 1.0
    x = int(round(float_num * (2 ** width)))
    print(f"Slope of {float_num} -> {x / (2 ** width)} (INT: {x})")
    return x


if __name__ == "__main__":

    DEFAULT_CONFIG = {
        "DATA_IN_0_PRECISION_0": 8,
        "DATA_IN_0_PRECISION_1": 2,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": 4,
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 2,
        "DATA_OUT_0_PRECISION_0": 8,
        "DATA_OUT_0_PRECISION_1": 2,
        "DATA_OUT_0_TENSOR_SIZE_DIM_0": 4,
        "DATA_OUT_0_TENSOR_SIZE_DIM_1": 4,
        "DATA_OUT_0_PARALLELISM_DIM_0": 2,
        "DATA_OUT_0_PARALLELISM_DIM_1": 2,
        "SLOPE": 1,
        "POWER_OF_2_SLOPE": 2,
    }

    mase_runner(
        module_param_list=[
            DEFAULT_CONFIG,
            {
                **DEFAULT_CONFIG,
                "POWER_OF_2_SLOPE": 4
            },
            {
                **DEFAULT_CONFIG,
                "SLOPE": 2,
                "FIXED_SLOPE": get_closest_fixed_slope(0.01, 8)
            },
            {
                **DEFAULT_CONFIG,
                "SLOPE": 2,
                "FIXED_SLOPE": get_closest_fixed_slope(0.124, 8)
            },
        ],
        # trace=True,
        # seed=1707332282
    )
