from os import path, getenv
import logging
from shutil import rmtree
from pathlib import Path
import re
import inspect
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

import torch

import cocotb
from cocotb.runner import get_runner, get_results
from mase_components.deps import MASE_HW_DEPS

logger = logging.getLogger("mase_runner")
logger.setLevel("INFO")


def mase_runner(
    module_param_list: list[dict[str, Any]] = [dict()],
    extra_build_args: list[str] = [],
    trace: bool = False,
    seed: int = None,
    jobs: int = 1,
):
    assert type(module_param_list) == list, "Need to pass in a list of dicts!"

    start_time = time()

    # Get file which called this function
    # Should be of form components/<group>/test/<module>_tb.py
    test_filepath = inspect.stack()[1].filename
    matches = re.search(r"mase_components/(\w*)/test/(\w*)_tb\.py", test_filepath)
    assert matches != None, "Function only works when called from test"
    group, module = matches.groups()

    # Group path is components/<group>
    group_path = Path(test_filepath).parent.parent

    # Components path is components/
    comp_path = group_path.parent

    # Try to find RTL file:
    # components/<group>/rtl/<module>.py
    module_path = group_path.joinpath("rtl").joinpath(f"{module}.sv")
    assert path.exists(module_path), f"{module_path} does not exist."

    SIM = getenv("SIM", "verilator")

    deps = MASE_HW_DEPS[f"{group}/{module}"]

    total_tests = 0
    total_fail = 0

    failed_cfgs = []

    def _single_test(
        test_work_dir: Path,
        module_params: dict,
        silent=False,
    ):
        cocotb.log
        if not silent:
            print("# ---------------------------------------")
            print(f"# Test {i+1}/{len(module_param_list)}")
            print("# ---------------------------------------")
            print(f"# Parameters:")
            print(f"# - {'Test Index'}: {i}")
            for k, v in module_params.items():
                print(f"# - {k}: {v}")
            print("# ---------------------------------------")

        runner = get_runner(SIM)
        runner.build(
            verilog_sources=[module_path],
            includes=[str(comp_path.joinpath(f"{d}/rtl/")) for d in deps],
            hdl_toplevel=module,
            build_args=[
                # Verilator linter is overly strict.
                # Too many errors
                # These errors are in later versions of verilator
                "-Wno-GENUNNAMED",
                "-Wno-WIDTHEXPAND",
                "-Wno-WIDTHTRUNC",
                # Simulation Optimisation
                "-Wno-UNOPTFLAT",
                "-prof-c",
                "--stats",
                # Signal trace in dump.fst
                *(["--trace-fst", "--trace-structs"] if trace else []),
                *(["--quiet-exit"] if silent else []),
                "-O2",
                "-build-jobs",
                "8",
                "-Wno-fatal",
                "-Wno-lint",
                "-Wno-style",
                *extra_build_args,
            ],
            parameters=module_params,
            build_dir=test_work_dir,
        )
        runner.test(
            hdl_toplevel=module,
            test_module=module + "_tb",
            seed=seed,
            results_xml="results.xml",
        )
        num_tests, fail = get_results(test_work_dir.joinpath("results.xml"))
        return {
            "num_tests": num_tests,
            "failed_tests": fail,
            "params": module_params,
        }

    # Single threaded run
    if jobs == 1:

        for i, module_params in enumerate(module_param_list):
            test_work_dir = group_path.joinpath(f"test/build/{module}/test_{i}")
            results = _single_test(
                test_work_dir=test_work_dir,
                module_params=module_params,
                silent=False,
            )
            total_tests += results["num_tests"]
            total_fail += results["failed_tests"]
            if results["failed_tests"]:
                failed_cfgs.append((i, module_params))

    # Multi threaded run
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:

            # Launch concurrent futures
            # TODO: add timeout
            future_to_id = {}
            for i, module_params in enumerate(module_param_list):
                test_work_dir = group_path.joinpath(f"test/build/{module}/test_{i}")
                future = executor.submit(_single_test, test_work_dir, module_params, True)
                future_to_id[future] = i

            # Wait for futures to complete
            for future in as_completed(future_to_id):
                id = future_to_id[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (id, exc))
                else:
                    print("Test %r is done. Result: %s" % (id, result))
                    total_tests += result["num_tests"]
                    total_fail += result["failed_tests"]
                    if result["failed_tests"]:
                        failed_cfgs.append((i, module_params))

    print("# ---------------------------------------")
    print("# Test Results")
    print("# ---------------------------------------")
    print("# - Time elapsed: %.2f seconds" % (time() - start_time))
    print("# - Jobs: %d" % (jobs))
    print("# - Passed: %d" % (total_tests - total_fail))
    print("# - Failed: %d" % (total_fail))
    print("# - Total : %d" % (total_tests))
    print("# ---------------------------------------")

    if len(failed_cfgs):
        print(f"# Failed Configs")
        print("# ---------------------------------------")
        for i, params in failed_cfgs:
            print(f"# - test_{i}: {params}")
        print("# ---------------------------------------")

    return total_fail


def simulate_pass(
    project_dir: Path,
    module_params: dict[str, Any] = {},
    extra_build_args: list[str] = [],
    trace: bool = False,
):
    rtl_dir = project_dir / "hardware" / "rtl"
    sim_dir = project_dir / "hardware" / "sim"
    test_dir = project_dir / "hardware" / "test" / "mase_top_tb"

    SIM = getenv("SIM", "verilator")

    runner = get_runner(SIM)
    runner.build(
        verilog_sources=[rtl_dir / "top.sv"],
        includes=[rtl_dir],
        hdl_toplevel="top",
        build_args=[
            # Verilator linter is overly strict.
            # Too many errors
            # These errors are in later versions of verilator
            "-Wno-GENUNNAMED",
            "-Wno-WIDTHEXPAND",
            "-Wno-WIDTHTRUNC",
            # Simulation Optimisation
            "-Wno-UNOPTFLAT",
            # Signal trace in dump.fst
            *(["--trace-fst", "--trace-structs"] if trace else []),
            "-prof-c",
            "--stats",
            "-O2",
            "-build-jobs",
            "8",
            "-Wno-fatal",
            "-Wno-lint",
            "-Wno-style",
            *extra_build_args,
        ],
        parameters=module_params,
        build_dir=sim_dir,
    )
    runner.test(
        hdl_toplevel="top",
        test_module="test",
        results_xml="results.xml",
    )
