#!/bin/bash

# Experiments for ECC submission.
SYS='quadrotor_3D'
TASK='tracking'
SYS_NAME='quadrotor'

# Run PPO-RL


ALGO='mpc_acados_m'
python3 ./ecc_sub_experiments.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}_${TASK}.yaml

# Run RTI
ALGO='mpc_acados'
CONFIG='rti'

python3 ./ecc_sub_experiments.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${CONFIG}_${SYS}_${TASK}.yaml

# Run Riccati-RTI
CONFIG='riccati_rti'

python3 ./ecc_sub_experiments.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${CONFIG}_${SYS}_${TASK}.yaml

# Run Riccati-RL
CONFIG='riccati_rl'

python3 ./ecc_sub_experiments.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${CONFIG}_${SYS}_${TASK}.yaml
