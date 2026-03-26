# Policy Composition for 2D Quadrotor Tracking

This repository contains the code used for the control experiment described in the paper, where two control primitives are composed to perform 2D quadrotor trajectory tracking.

The implementation is built on top of [Safe-Control-Gym](https://github.com/utiasDSL/safe-control-gym), using its Crazyflie quadrotor model, PID controller implementation, and trajectory-tracking environment.

## Overview

We consider a noisy 2D quadrotor tracking task with a lemniscate reference trajectory from `Safe-Control-Gym`. The state and control are affected by Gaussian disturbances.

Two primitive controllers are available:

- **Stabilizing PD primitive**: regulates the quadrotor attitude and damps motion so that the drone remains near its current position.
- **Tracking PD primitive**: attempts to follow the reference trajectory, but with gains that are insufficient for accurate curvature tracking.


## Files

- `PD-PD_control.py`  
  Runs the experiment, evaluates all controllers, and saves the results.

- `plots_PD-PD.py`  
  Loads the saved results and generates the figures used for analysis.

## Dependency

This code is designed to be run **inside the root directory** of a local clone of:

- [Safe-Control-Gym](https://github.com/utiasDSL/safe-control-gym)

The scripts use modules and configuration files from that project, including:

- `safe_control_gym.utils.registration`
- `safe_control_gym.utils.configuration`
- `safe_control_gym.controllers.pid.pid`
- `safe_control_gym.controllers.lqr.lqr_utils`
- `examples/pid/config_overrides/quadrotor_2D/quadrotor_2D_stab.yaml`
- `examples/pid/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml`

## Setup

Clone and install `Safe-Control-Gym` following its official instructions. Then place the two files

- `PD-PD_control.py`
- `plots_PD-PD.py`

in the root folder of the cloned repository, e.g.

```text
safe-control-gym/
├── safe_control_gym/
├── examples/
├── PD-PD_control.py
├── plots_PD-PD.py
└── ...
```

## Requirements
```
numpy>=1.23
matplotlib>=3.6
scipy>=1.9
pybullet>=3.2
```

## Running the experiment
From the root of safe-control-gym, run:

`python PD-PD_control.py`

This script:
- creates the noisy 2D quadrotor tracking environment,
- evaluates the policy composition method,
- evaluates the two individual PD primitives,
- evaluates the PID baseline,
- saves the results to:
  `Results_PD-PD/experiment_data.npz`

## Generating the plots
After running the experiment, generate the figures with:

`python plots_PD-PD.py`

The script produces:
- total episode cost plot
- trajectory overlay figure comparing the composed controller and the PID baseline

The figures are saved in:
`Results_PD-PD/Figures/`


