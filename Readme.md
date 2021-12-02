# Pretrained Cost Model for Distributed Constraint Optimization Problems

# Requirements
- **PyTorch 1.9.0**
- **PyTorch Geometric 1.7.1**

# Directory structure

- `baselines` contains the implementation of all compared baselines
- `core` contains the core data structures to run the simulation
- `entry` contains the entry point of each algorithm
- `heuristics` contains the implementation of  GAT-PCM-boosted algorithms
- `pretrain` contains the implementation of pretraining stage

# How to run the code

See the command line interface of `run_*.py` in `entry`.

Example:

`python -um entry.run_dsa -pth problem.xml -c 1000 -p 0.8`