# Practical 2 (ddl20)


## Lab 3
### 1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

I have implemented these extra metrics:
- FLOPs by using ptflops
- BitOPs

They both have their own [hardware runners](../machop/chop/actions/search/strategies/runners/hardware/) which call into their respective analysis passes in [flop_estimator](../machop/chop/passes/graph/analysis/flop_estimator/ptflops.py) and [quantization](../machop/chop/passes/graph/analysis/quantization/calculate_bitops.py).

### 2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It's important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).

I have combined the metrics described above with the accuracy quality metric and
the full `.toml` config can be seen [here](../coursework/lab3/jsc_my_search.toml).

The scale factors have been assigned to each of the metrics as shown:
- accuracy: 1.0
- average_bitwidth: 0.2
- flops: 0.1
- bitops: 0.2

Accuracy is assigned the highest metric as I believe that is the main objective
of the network architecture search, while other metrics are more related to
computational complexity or hardware/memory usage. Essentially, a model which
has a terrible accuracy is useless no matter how computationally efficient it is.

Regarding the assignment of scale for bitops, flops and average_bitwidth, I
believe that the bitops and average_bitwidth metrics capture most of the computational
complexity as well as memory usage.

### 3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

Brute force search has been implemented [here](../machop/chop/actions/search/strategies/brute_force.py). It takes the cross product of all input configurations and runs them one at a time to find the config with the best score. The calcuation of the score is defined through the `.toml` file.

### 4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.

The final neural network performance of the brute-force search will always be better than the TPE based search as it is able to fully explore the discrete parameter space you give it and arrive at a global maxima for your score. On the contrary, the TPE is a search method which uses a history of evaluated hyperparameters to suggest the next set of hyperparameters to test. TPE search may iteratively arrive at a local maxima when it exhausts all iterations, however there is no guarantee that this is the global one.

However, the brute-force approach comes at a cost as the number of search iterations will increase exponentially with the number of parameters you are searching through. This means it becomes computationally unfeasable to search through a space for larger models with many layers unless you drastically reduce the number of parameters.

## Lab 4

I have chosen to implement [LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU) as my PyTorch layer.

My implementation of LeakyReLU allows for two different ways to parameterise the
slope when the values passed in are negative:
1. A hardware friendly version which means that LeakyReLU can only have slopes of the form $2^{-n}, n \in \Z$.
2. An arbitrary fixed point fractional number of the same width as the input.

Having two different implementations or essentially quantisations for the slope, the user can better decide how they want to tradeoff latency and resource usage vs. accuracy. Both implementations do the LeakyReLU operation combinatorially, however, the one which only allows slope gradients of the form $2^{-n}$ is essentially free as it can be implemented as a simple bitshift, while the fixed point implementation will use a multiplier which will also add substantial delay to the combinatorial path if this block is not pipelined. In terms of accuracy, the LeakyReLU which only supports gradients of form $2^{-n}$ may perform worse than the fixed point if post traning quantization (PTQ) is performed, however, if the original network is trained with the LeakyReLU set to a power of two, this loss in accuracy can be prevented all together.

The SystemVerilog implementation is in [mase_components/activations/rtl/fixed_leaky_relu.sv](../machop/mase_components/activations/rtl/fixed_leaky_relu.sv), and the cocotb test is at [mase_components/activations/test/fixed_leaky_relu_tb.py](../machop/mase_components/activations/test/fixed_leaky_relu_tb.py).
