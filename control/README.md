# Policy Composition for Quadrotor Control
This folder contains the Python code used for the control experiments described in the paper and the SI Appendix.
Two control primitives are composed to perform a trajectory tracking task for a 2D quadrotor.

The implementation is built on top of [Safe-Control-Gym](https://github.com/utiasDSL/safe-control-gym), using its Crazyflie quadrotor model, controller implementations, and trajectory-tracking environment.

## Overview

We consider a noisy 2D quadrotor tracking task with a lemniscate reference trajectory from `Safe-Control-Gym`.

We study two controller-composition settings:

- **PD-PD composition**: This experiment is described in the Results section of the main text. Two PD controllers are composed. The state and control input are affected by Gaussian disturbances.
  - **Stabilizing PD primitive**: regulates the quadrotor attitude and damps motion so that the drone remains near its current position.
  - **Tracking PD primitive**: attempts to follow the reference trajectory, but with gains that are insufficient for accurate curvature tracking.

- **LQR-LQR composition**: This experiment is described in Section 5 of the SI Appendix. Two LQR controllers are composed. The state and control input are affected by Gaussian disturbances. Moreover, the state is also affected by an impulse disturbance.
  - **Stabilizing LQR primitive**: regulates the quadrotor around its current position using an LQR controller configured for stabilization.
  - **Tracking LQR primitive**: attempts to follow the reference trajectory using an LQR controller configured for trajectory tracking.

In both cases, the experiment evaluates:
- our policy composition method,
- the two individual primitives,
- and a baseline controller (PID for PD-PD, LQR for LQR-LQR).

## Files

- `PD-PD_control.py`  
  Runs the experiment for composition of two PD primitives, evaluates all controllers, and saves the results.

- `plots_PD-PD.py`  
  Loads the saved PD-PD results and generates the figures used for analysis.

- `LQR-LQR_control.py`  
  Runs the experiment for composition of two LQR primitives, evaluates all controllers, and saves the results.

- `plots_LQR-LQR.py`  
  Loads the saved LQR-LQR results and generates the figures used for analysis.

## Dependencies

This code is designed to be run **inside the root directory** of a local clone of:

- [Safe-Control-Gym](https://github.com/utiasDSL/safe-control-gym)

The scripts use modules and configuration files from that project, including:

- `safe_control_gym.utils.registration`
- `safe_control_gym.utils.configuration`
- `safe_control_gym.controllers.pid.pid`
- `safe_control_gym.controllers.lqr.lqr`
- `safe_control_gym.controllers.lqr.lqr_utils`
- `examples/pid/config_overrides/quadrotor_2D/quadrotor_2D_stab.yaml`
- `examples/pid/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml`
- `examples/lqr/config_overrides/quadrotor_2D/quadrotor_2D_stab.yaml`
- `examples/lqr/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml`
- `examples/lqr/config_overrides/quadrotor_2D/lqr_quadrotor_2D_stab.yaml`
- `examples/lqr/config_overrides/quadrotor_2D/lqr_quadrotor_2D_track.yaml`

## Setup

Clone and install `Safe-Control-Gym` following its official instructions. Then place the experiment scripts in the root folder of the cloned repository, e.g.

```text
safe-control-gym/
├── safe_control_gym/
├── examples/
├── PD-PD_control.py
├── plots_PD-PD.py
├── LQR-LQR_control.py
├── plots_LQR-LQR.py
└── ...
````

## Requirements

```text
numpy>=1.23
matplotlib>=3.6
scipy>=1.9
pybullet>=3.2
```

## Running the experiments

### PD-PD composition

From the root of `safe-control-gym`, run:

```bash
python PD-PD_control.py
```

This script:

* creates the noisy 2D quadrotor tracking environment,
* evaluates the policy composition method,
* evaluates the two individual PD primitives,
* evaluates the PID baseline,
* saves the results to:

```text
Results_PD-PD/experiment_data.npz
```

### LQR-LQR composition

From the root of `safe-control-gym`, run:

```bash
python LQR-LQR_control.py
```

This script:

* creates the noisy 2D quadrotor tracking environment,
* evaluates the policy composition method,
* evaluates the two individual LQR primitives,
* evaluates the LQR baseline,
* saves the results to:

```text
Results_LQR-LQR/experiment_data.npz
```

## Generating the plots

### PD-PD plots

After running the PD-PD experiment, generate the figures with:

```bash
python plots_PD-PD.py
```

The script produces:

- total episode cost plot
  
  <a href="https://github.com/user-attachments/files/26264852/total_episode_cost.pdf">
  <img src="https://github.com/user-attachments/assets/119a0021-52ab-4103-8de8-a3cce5472ca8" width="600">
  </a>

- figure comparing the trajectories of the drone controlled by our policy composition method and the PID baseline

  <a href="https://github.com/user-attachments/files/26264957/traj_our_pid.pdf">
  <img src="https://github.com/user-attachments/assets/b7031a52-1acc-4040-aaff-091359f57b53" width="600">
  </a>

The figures are saved in:

```text
Results_PD-PD/Figures/
```

### LQR-LQR plots

After running the LQR-LQR experiment, generate the figures with:

```bash
python plots_LQR-LQR.py
```

The script produces:
- total episode cost plot

  <a href="https://github.com/user-attachments/files/26662419/total_episode_cost.pdf">
  <img src="https://github.com/user-attachments/assets/178d9624-4d26-4149-aa8a-8c40fa5b03c9" width="600">
  </a>

- figure comparing the trajectories of the drone controlled by our policy composition method and the LQR baseline

  <a href="https://github.com/user-attachments/files/26662420/traj_ours_lqr.pdf">
  <img src="https://github.com/user-attachments/assets/a8956f39-90b7-4d2d-901a-697999a716f4" width="600">
  </a>

The figures are saved in:

```text
Results_LQR-LQR/Figures/
```


## Acknowledgment

This code builds on the `Safe-Control-Gym` framework:

Z. Yuan, A. W. Hall, S. Zhou, L. Brunke, M. Greeff, J. Panerati, and A. P. Schoellig, "Safe-Control-Gym: A Unified Benchmark Suite for Safe Learning-Based Control and Reinforcement Learning in Robotics", *IEEE Robotics and Automation Letters*, (2022).

Official repository:
[Safe-Control-Gym](https://github.com/utiasDSL/safe-control-gym)





