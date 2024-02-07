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

    // Possible Slope Choices:
    // 1 - Power of 2:
    //     LeakyReLU.negative_slope = 1 / 2^(negative_slope_exp)
    // 2 - Fixed Point Number
    //     LeakyReLU.negative_slope = slope
    //     Where: slope is a unsigned fixed point number which same width as DATA_IN, but
    //     it has frac bits equal to the width. This means it can only represent
    //     numbers between 0 - 1.
    parameter SLOPE = 1,

    // torch.LeakyReLU.negative_slope = 1 / 2^(negative_slope_exp)
    parameter POWER_OF_2_SLOPE = -1,
    parameter logic [DATA_IN_0_PRECISION_0-1:0] FIXED_SLOPE = 0
) (
    input logic rst,
    input logic clk,

    input logic signed [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input  logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic signed [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
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
    assert (SLOPE == 1 || SLOPE == 2);

    // Not going to handle rounding, can use cast module if you choose rounding
    assert (DATA_IN_0_PRECISION_0 == DATA_OUT_0_PRECISION_0 &&
            DATA_IN_0_PRECISION_1 == DATA_OUT_0_PRECISION_1);
end

// Internal
localparam logic signed [DATA_IN_0_PRECISION_0:0] SIGNED_FIXED_SLOPE = {1'b0, FIXED_SLOPE};

for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++) begin : LeakyReLU
    always_comb begin
        // negative value, put to zero
        if (data_in_0[i] < 0) begin
            if (SLOPE == 1) begin
                data_out_0[i] = (data_in_0[i] >>> POWER_OF_2_SLOPE);
            end else if (SLOPE == 2) begin
                logic signed [2*DATA_IN_0_PRECISION_0-1:0] mult_res;
                mult_res = (data_in_0[i] * SIGNED_FIXED_SLOPE);
                data_out_0[i] = mult_res >>> DATA_IN_0_PRECISION_0;
            end
        end else begin
            data_out_0[i] = data_in_0[i];
        end
    end
end

assign data_out_0_valid = data_in_0_valid;
assign data_in_0_ready  = data_out_0_ready;

endmodule
