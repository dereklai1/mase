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

    // torch.LeakyReLU.negative_slope = 1 / 2^(negative_slope_exp)
    parameter int NEGATIVE_SLOPE_EXP = 2
) (
    input logic rst,
    input logic clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++) begin : LeakyReLU
    always_comb begin
        // negative value, put to zero
        if ($signed(data_in_0[i]) <= 0) begin
            data_out_0[i] = (data_in_0[i] >>> NEGATIVE_SLOPE_EXP);
        end else begin
            data_out_0[i] = data_in_0[i];
        end
    end
end

endmodule
