# Practical 2 (ddl20)


## Lab 3
### 1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

I have implemented these extra metrics:
- FLOPs by using ptflops
- BitOPs
<!-- - Latency Estimation -->

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

### 4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.
