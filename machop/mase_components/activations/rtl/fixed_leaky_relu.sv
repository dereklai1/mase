`timescale 1ns / 1ps

module fixed_leaky_relu #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 0,

    // Possible types:
    // "power-of-2":
    // "fixed": Uses a fixed point number with
    parameter SLOPE_TYPE = "power-of-2",

    // torch.LeakyReLU.negative_slope = 1 / 2^(negative_slope_exp)
    parameter int POWER_OF_2_SLOPE = 2
) (
    input logic rst,
    input logic clk,

    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input  logic data_in_0_valid,
    output logic data_in_0_ready,'

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);


initial begin : Assertions
    // Assert that input dimensions match up, not going to support reshaping
    assert (DATA_IN_0_TENSOR_SIZE_DIM_0 == DATA_OUT_0_TENSOR_SIZE_DIM_0 &&
            DATA_IN_0_TENSOR_SIZE_DIM_1 == DATA_OUT_0_TENSOR_SIZE_DIM_1 &&
            DATA_IN_0_PARALLELISM_DIM_0 == DATA_OUT_0_TENSOR_SIZE_DIM_0 &&
            DATA_IN_0_PARALLELISM_DIM_1 == DATA_OUT_0_TENSOR_SIZE_DIM_1);

    // Assert that a valid leaky slope type is chosen
    assert (LEAKY_TYPE == "power-of-2" || LEAKY_TYPE == "fixed");

    // Not going to handle rounding, can use cast module if you choose rounding
    assert (DATA_IN_0_PRECISION_0 == DATA_OUT_0_PRECISION_0 &&
            DATA_IN_0_PRECISION_1 == DATA_OUT_0_PRECISION_1);
end


for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++) begin : LeakyReLU
    always_comb begin
        // negative value, put to zero
        if ($signed(data_in_0[i]) <= 0) begin
            data_out_0[i] = (data_in_0[i] >>> POWER_OF_2_SLOPE);
        end else begin
            data_out_0[i] = data_in_0[i];
        end
    end
end

endmodule
