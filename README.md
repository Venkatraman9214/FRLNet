# FRLNet Policy Manager

MATLAB implementation of the **FRLNet Policy Manager Algorithm**, a hybrid reinforcement learning framework for queue-aware scheduling and resource-efficient decision making in UAV networks.

---

## Overview

This repository provides a simulation of the FRLNet Policy Manager Algorithm, which combines:

- Model-Free Reinforcement Learning (Exploration)
- Model-Based Reinforcement Learning (Exploitation)
- Queue Stabilization
- Markov Decision Process (MDP) Optimization
- Adaptive Exploration-Exploitation Scheduling

The objective is to minimize aggregate queue backlog while learning efficient scheduling decisions in dynamic communication environments.

---

## Key Features

### Hybrid Reinforcement Learning

FRLNet combines two complementary learning mechanisms:

#### Exploration Phase (Model-Free Learning)

- Interacts directly with the environment
- Collects state-action-transition samples
- Requires no prior knowledge of system dynamics
- Uses randomized actions for learning

#### Exploitation Phase (Model-Based Learning)

- Estimates transition probabilities from collected samples
- Constructs an empirical MDP model
- Solves the MDP using Value Iteration
- Applies the learned policy to reduce congestion

---

## Queueing System Model

The simulator models:

- Multiple queues (currently two queues)
- Bernoulli packet arrivals
- Probabilistic packet service
- Discrete-time operation
- Dynamic queue evolution

The optimization objective is:

\[
C_f(V) = \sum_i Q_i
\]

where:

- \(Q_i\) is the backlog of queue \(i\)
- \(V\) is the system backlog state

---

## State Representation

The system state is represented by

\[
V(t) = [Q_1(t), Q_2(t)]
\]

where:

- \(Q_i(t)\) denotes the queue length of queue \(i\) at time slot \(t\)

The maximum queue backlog is

\[
V_{\max}(t)=\max_i Q_i(t)
\]

---

## Learning Process

### Step 1: Exploration

A randomized policy

\[
\beta_{rnd}
\]

collects state-transition samples.

### Step 2: Transition Estimation

Observed transitions are used to estimate

\[
\hat{\pi}(V'|V,a)
\]

### Step 3: MDP Construction

A truncated MDP is constructed using the learned transition model.

### Step 4: Value Iteration

The estimated MDP is solved to obtain

\[
\bar{\beta}^{*}
\]

### Step 5: Policy Execution

The learned policy is applied in the stable operating region while a baseline stabilizing policy is used elsewhere.

---

## Event-Driven Episodes

Unlike fixed-length RL episodes, FRLNet uses event-driven episodes.

The exploration threshold is defined as:

\[
\zeta_{\theta} = \frac{r}{\sqrt{\theta}}
\]

where:

- \(r\) is the exploration-decay parameter
- \(\theta\) is the episode index

Exploration gradually decreases as learning progresses.

---

## Default Simulation Parameters

| Parameter | Value |
|------------|--------|
| Queue 1 Arrival Rate (\(\lambda_1\)) | 0.35 packets/slot |
| Queue 2 Arrival Rate (\(\lambda_2\)) | 0.25 packets/slot |
| Queue 1 Service Success (\(\pi_1\)) | 0.60 |
| Queue 2 Service Success (\(\pi_2\)) | 0.85 |
| Queue Truncation Threshold (T) | 10 |
| Exploration Parameter (r) | 0.5 |
| Episode Threshold Parameter (R) | 10 |
| Simulation Horizon | 10,000 time slots |

---

## Outputs

The simulator generates:

### Aggregate Queue Backlog

Tracks:

\[
\sum_i Q_i
\]

over time.

### Learning Statistics

- Episode lengths
- Exploration vs exploitation behavior
- Policy evolution

### High-Resolution Figures

All figures are exported at **600 DPI**, suitable for publication.

---

## Running the Simulator

Open MATLAB and run:

```matlab
frlnet_policy_manager_sim
```

---

## Repository Structure

```text
.
├── frlnet_policy_manager_sim.m
├── figures/
├── results/
└── README.md
```

---

## Research Applications

This simulator can be used for studying:

- UAV Networking
- Queue-Aware Reinforcement Learning
- Resource-Constrained Edge Systems
- Distributed Scheduling
- Frugal Reinforcement Learning
- MDP-Based Optimization

---

## Future Work

Planned extensions include:

- Multi-Agent Cooperative Learning
- Hardware-in-the-Loop UAV Testbeds
- Energy-Aware Scheduling
- Multi-Hop Routing

---

## License

This repository is released for research and educational use.

---

## Citation

If you use this simulator in your research, please cite the associated FRLNet publication.

```bibtex
@article{FRLNet,
  title={FRLNet: A Frugal Reinforcement Learning Framework for Resource-Constrained Networks},
  author={V.~Balasubramanian et. al},
  year={2026}
}
```
