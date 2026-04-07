#!/bin/bash

# MPC and Linear MPC Experiment.

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_3D'

# TASK='stabilization'
TASK='tracking'

# ALGO='mpc'
ALGO='mpc_acados'
# ALGO='mpc_acados_m'
# ALGO='linear_mpc'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

CONFIG='riccati_rti'  # riccati_rl  riccati_rti, rti

python3 ./mpc_experiment.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${CONFIG}_${SYS}_${TASK}.yaml
