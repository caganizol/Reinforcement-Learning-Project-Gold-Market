[README_EEE448.md](https://github.com/user-attachments/files/29109630/README_EEE448.md)
# Dynamic Programming and Reinforcement Learning for a Finite Trading MDP

This repository contains the MATLAB implementation and project report for **EEE 448 - Dynamic Programming and Reinforcement Learning**, focusing on **Tasks 3 and 4**.

The project solves a finite tabular trading Markov Decision Process (MDP) using:

- Value Iteration
- Policy Iteration
- Model learning followed by Dynamic Programming
- Tabular Q-learning
- Tabular Policy Gradient / Actor-Critic

The main objective is to compare exact dynamic programming methods with model-based and model-free reinforcement learning approaches for a finite trading problem.

## Project Context

The project uses a finite tabular trading MDP with state:

```text
s = (x, B, m, μ)
```

where:

- `x` is the asset price
- `B` is the liquid budget
- `m` is the number of asset holdings
- `μ` is the market regime: Bear, Sideways, or Bull

The initial state is:

```text
x0 = 10
B0 = 30
m0 = 0
μ0 = Sideways
```

The numerical setup is:

```text
Price cap:        x_bar = 20
Budget cap:       Bmax = 50
Holdings cap:     mmax = 10
Discount factor:  gamma = 0.95
State-space size: 21 × 51 × 11 × 3 = 35,343 states
```

## Repository Contents

```text
.
├── main.m                 # MATLAB implementation of Tasks 3 and 4
├── report.pdf             # Project report
├── figures_task3_task4/   # Generated figures and CSV tables, created after running the MATLAB script
└── README.md
```

> Depending on your file names, `main.m` and `report.pdf` may have different names in the repository.

## Implemented Methods

### 1. Value Iteration

Value iteration is used to solve the known-model discounted infinite-horizon MDP.

Reported result:

```text
Value iteration sweeps: 284
Final Bellman residual: 9.54e-9
V*(s0): 14.2808
Optimal action at s0: hold
```

### 2. Policy Iteration

Policy iteration starts from the hold policy and alternates between policy evaluation and policy improvement.

Reported result:

```text
Policy improvement sweeps: 6
Final V(s0): 14.2808
Policy agreement with value iteration: 100%
```

### 3. Learned-Model Dynamic Programming

Instead of using the known transition model directly, the transition kernel is estimated from simulator-generated samples.

After learning the model, value iteration is applied to the estimated MDP.

Sample sizes tested:

```text
2,000
10,000
50,000
200,000
```

With 200,000 samples, the learned-model DP policy nearly matches the known-model DP policy.

### 4. Q-Learning

Tabular Q-learning is implemented as a model-free off-policy temporal-difference control method.

Key settings:

```text
Training episodes: 2,000,000
Training horizon: 250
Exploration: epsilon-greedy
Epsilon decay: 0.80 to 0.01
Learning rate decay: 0.08 to 0.005
Random starts: 20%
```

Q-learning learns a more conservative policy with a lower bankruptcy rate, but lower expected discounted return than dynamic programming.

### 5. Policy Gradient / Actor-Critic

A tabular actor-critic method is implemented as the policy-gradient approach.

Key settings:

```text
Training episodes: 2,000,000
Training horizon: 250
Policy: softmax over feasible actions
Temperature decay: 1.0 to 0.1
Actor learning rate: 0.0015 to 0.00005
Critic learning rate: 0.03 to 0.001
TD-error clipping: ±100
```

The actor-critic policy achieves lower performance than the dynamic programming methods and shows a higher bankruptcy rate.

## Final Performance Summary

| Method | Expected Return | Empirical Return | Bankruptcy Rate |
|---|---:|---:|---:|
| Known-model DP | 14.281 | 14.313 | 7.666% |
| Learned-model DP | 14.258 | 14.502 | 7.355% |
| Q-learning | 12.758 | 12.822 | 2.357% |
| Actor-Critic PG | 12.359 | 12.385 | 9.992% |

The known-model dynamic programming policy gives the highest expected discounted return. The learned-model DP approach performs almost identically when enough transition samples are used. Q-learning is more conservative and produces the lowest bankruptcy rate, while actor-critic has the highest bankruptcy rate among the compared methods.

## MATLAB Script Overview

The MATLAB script performs the following steps:

1. Defines MDP parameters and probability matrices
2. Builds state and feasible-action tables
3. Runs value iteration
4. Runs policy iteration
5. Simulates the optimal DP policy
6. Learns an exogenous transition model from samples
7. Repeats dynamic programming using the learned model
8. Trains a tabular Q-learning agent
9. Trains a tabular actor-critic policy-gradient agent
10. Compares all policies using simulation
11. Exports figures and CSV summary tables

## Generated Outputs

After running the MATLAB script, the following outputs are generated in the `figures_task3_task4` folder:

```text
fig1_value_iteration_convergence.png
fig2_policy_iteration_convergence.png
fig3_performance_comparison_with_learned_model.png
fig4_bankruptcy_comparison_with_learned_model.png
fig5_learned_model_sample_size.png
fig6_average_wealth_across_models.png
fig7_training_curves.png

table_learned_model_comparison.csv
table_overall_performance_comparison.csv
table_average_wealth_comparison.csv
```

## How to Run

1. Open MATLAB.
2. Place the `.m` file in your working directory.
3. Run the script:

```matlab
run main.m
```

or, if your file has a different name:

```matlab
run your_file_name.m
```

4. The script prints the results in the MATLAB Command Window.
5. Figures and CSV tables are saved automatically into the `figures_task3_task4` directory.

## Requirements

- MATLAB
- No additional toolbox is required for the core tabular implementation

## Main Results

The main conclusions are:

- Value iteration converges to the optimal solution in 284 sweeps.
- Policy iteration reaches the same optimal policy in 6 improvement sweeps.
- Known-model DP achieves the best expected discounted return.
- Learned-model DP nearly matches the known-model solution with enough samples.
- Q-learning learns a safer but more conservative policy.
- Actor-critic policy gradient is more variable and has a higher bankruptcy rate.

## Author

**Çağan İzol**  
Bilkent University  
Department of Electrical and Electronics Engineering  
EEE 448 - Dynamic Programming and Reinforcement Learning

## License

This repository is intended for educational and academic purposes.  
You may add a license such as MIT if you want others to freely use or modify the code.
