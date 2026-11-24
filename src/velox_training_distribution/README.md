# Velox training composition analysis

The population for this analysis is the set of edges consumed by a completed Velox training run. Start with that run's frozen configuration, transformed provenance graphs and saved training tensors. The native loader batches the tensors. Its training log and independently verified coverage report provide the expected counts.

For the CADETS E3 run `cadets-velox-fork-train-20261124a` (seed 0, commit `8a7303915422a85dead0a627ab1a031b7ef0c6b5`), the expected input contains 581 training windows, 3,426,589 edges and 3,665 batches. Bind a native edge to its window, tensor row and batch position before counting compositions. Restrict traversal to the graph windows supplied to Velox. Use the original graph's identifiers to trace source records. An audit-log event dropped by preprocessing is outside this population, even when its timestamp belongs to a training day.
