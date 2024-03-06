#!/usr/bin/env python3

# This script tests the fixed point inverse square root.
import random, os

import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner
import math

def float_to_int(x: float, int_width: int, frac_width: int) -> int:
    integer = int(x)
    x -= integer
    res = integer * (2 ** frac_width)
    for i in range(1, frac_width+1):
        power = 2 ** (-i)
        if power <= x:
            x -= power
            res += 2 ** (frac_width - i)
    return res

def int_to_float(x: int, int_width: int, frac_width: int) -> float:
    integer = x / (2 ** frac_width)
    fraction = x - integer * 2 ** frac_width
    res = integer

    for i in range(1, frac_width+1):
        power = 2 ** (frac_width - i)
        if power < fraction:
            res += 2 ** (-i)
            fraction -= power
    return res

def isqrt_sw(x: int, int_width: int, frac_width: int) -> int:
    """model of multiplier"""
    if x == 0:
        return 2 ** (int_width + frac_width) - 1
    x_f = int_to_float(x, int_width, frac_width)
    ref = 1 / math.sqrt(x_f)
    return ref

class VerificationCase:
    def __init__(self, samples=1):
        self.data_in_width = 16
        self.int_width = 8
        self.frac_width = 8
        self.data_in = [val for val in range(samples)]
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "INT_WIDTH": self.int_width,
            "FRAC_WIDTH": self.frac_width
        }

    def sw_compute(self):
        ref = []
        for sample in self.data_in:
            expected = isqrt_sw(sample, self.int_width, self.frac_width)
            ref.append(expected)
        return ref


@cocotb.test()
async def test_fixed_isqrt(dut):
    """Test for adding 2 random numbers multiple times"""
    samples = 2**16-1
    testcase = VerificationCase(samples=samples)

    #for i in range(samples):
    for i in range(samples):
        # Set up module data.
        data_a = testcase.data_in[i]

        # Force module data.
        dut.data_a.value = data_a

        # Wait for processing.
        await Timer(10, units="ns")

        # Exepected result.
        expected = testcase.ref[i]

        # Check the output.
        assert (
                int_to_float(dut.isqrt.value.integer, 8, 8) - expected < 2**(-8)
            ), f"""
            <<< --- Test failed --- >>>
            Input: 
            X  : {int_to_float(data_a, 8, 8)}

            Output:
            Out: {int_to_float(dut.isqrt.value.integer, 8, 8)}
            
            Expected: 
            {int_to_float(expected, 8, 8)}

            Test index:
            {i}
            """


if __name__ == "__main__":
    mase_runner()
