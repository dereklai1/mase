import cocotb
from cocotb.triggers import *
from cocotb.clock import Clock


class Testbench:
    __test__ = False  # so pytest doesn't confuse this with a test

    def __init__(self, dut, clk=None, rst=None, fail_on_checks=True) -> None:
        self.dut = dut
        self.clk = clk
        self.rst = rst

        self.input_drivers = []
        self.output_monitors = []

        self.input_precision = [32]

        self.fail_on_checks = fail_on_checks

        if self.clk != None:
            self.clock = Clock(self.clk, 20, units="ns")
            cocotb.start_soon(self.clock.start())

    def assign_self_params(self, attrs):
        for att in attrs:
            setattr(self, att, getattr(self.dut, att).value)

    async def reset(self, active_high=True):
        if self.rst == None:
            raise Exception(
                "Cannot reset. Either a reset wire was not provided or "
                + "the module does not have a reset."
            )

        await RisingEdge(self.clk)
        self.rst.value = 1 if active_high else 0
        await RisingEdge(self.clk)
        self.rst.value = 0 if active_high else 1
        await RisingEdge(self.clk)

    async def initialize(self):
        await self.reset()

        # Set all monitors ready
        for monitor in self.output_monitors:
            monitor.ready.value = 1

    def generate_inputs(self, batches=1):
        raise NotImplementedError

    def load_drivers(self, in_tensors):
        raise NotImplementedError

    def load_monitors(self, expectation):
        raise NotImplementedError

    def end_checks(self):
        if self.fail_on_checks:
            for monitor in self.output_monitors:
                assert monitor.exp_queue.empty()

            for driver in self.input_drivers:
                assert driver.send_queue.empty()
