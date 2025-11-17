# Riccati-RL: Refining Learned Policy Rollouts via a Single Riccati Recursion in Real-Time Iteration Model Predictive Control
This repo is linked with the [paper] (*submitted for publication*) where we propose to compute a single Newton step over the rollout obtained via a learned policy.

The repo contains the code to reproduce the example contained in the paper.
The example is constructed within [`safe-control-gym`](https://github.com/utiasDSL/safe-control-gym) and consists of tracking a lemniscate with a nonlinear 3D quadcopter.
We adopt as learned policy the PPO policy the one available in the `safe-control-gym` library.
Installation instructions are provided below.

![alt text](https://github.com/aghezz1/rl-riccati/blob/resub/figures/pos_projections.png)
![alt text](https://github.com/aghezz1/rl-riccati/blob/resub/figures/vel_x_z_subplots.png)

## Usage
After installation, to run the simulations in the paper need move to `cd examples/mpc` and execute `./ecc_sub_experiments.sh` \
To reproduce the plots, from the root folder, run the python script `python examples/results_analysis_onephase.py` \
_Note:_ you might need to edit the parameter `DATE` inside `results_analysis_onephase.py` since the results are saved by date

## Install on Ubuntu/macOS

### Clone repo

```bash
git clone https://github.com/aghezz1/rl-riccati
cd pept
```

### Create a `conda` environment

Create and access a Python 3.10 environment using
[`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n safe python=3.10
conda activate safe
```

### Install `rl-riccati` repo

#### Install `safe-control-gym`

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

#### Install `L4CasADi` (CPU only installation)

Ensure all build dependencies are installed
```
setuptools>=68.1
scikit-build>=0.17
cmake>=3.27
ninja>=1.11
```
Run
```bash
python -m pip install l4casadi --no-build-isolation
```
**NOTE**: I experienced issues with the compilation of CasADi functions on macOS as the default compiler in l4casadi is `gcc`, I solved by aliasing `gcc` to `clang`.
Another option is to open the l4casadi package installed in your environment and edit [L337](https://github.com/Tim-Salzmann/l4casadi/blob/eb6fc5c81aee29340b7e4b96e71226a88e1fa54c/l4casadi/l4casadi.py#L337) from `gcc` to `clang`.

More info at [l4casadi github](https://github.com/Tim-Salzmann/l4casadi)

#### Install `acados`

You need to separately install [`acados`](https://github.com/acados/acados) (>= v0.4.4) for fast MPC implementations.

- To build and install acados, see their [installation guide](https://docs.acados.org/installation/index.html).
- To set up the acados python interface **in the same conda environment!**, check out [these installation steps](https://docs.acados.org/python_interface/index.html).
  ```bash
  python -m pip install -e PATH_TO_ACADOS_DIR/interfaces/acados_template
  ```


