#!/usr/bin/env bash
#SBATCH --job-name=Option
#SBATCH --cpus-per-task=16   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=256G        # memory per node
#SBATCH --time=0-06:00      # time (DD-HH:MM)
#SBATCH --output=logs_options/%x_%A_%a.out
#SBATCH --error=logs_options/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-20


set -euo pipefail

# Move into repo
cd ~/scratch/MDP-Experiment-Framework

# Load modules & env
# module python/3.10
module load mujoco
export MUJOCO_GL=egl
# source ~/ENV/bin/activate
source ~/scratch/envs/venv2/bin/activate

# Pin BLAS/OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

# Compute array‐task index
IDX=$SLURM_ARRAY_TASK_ID   

# --------Random Variables-------
NUM_DISTRACTORS=15

ENVIRONMENT_SEEDS=(10 12 20 44 49)

declare -A WORKING_SEEDS

WORKING_SEEDS[10]="$(seq 0 49)"
WORKING_SEEDS[12]="1 2 3 5 6 7 8 14 15 16 23 28 29 30 32 33 34 36 37 39 42 47"
WORKING_SEEDS[20]="$(seq 0 49)"
WORKING_SEEDS[30]=""
WORKING_SEEDS[37]=""
WORKING_SEEDS[44]="2 3 7 8 9 10 11 12 16 17 18 19 22 23 24 31 32 33 35 39 40 42 44 45 49"
WORKING_SEEDS[49]="9 11 14 15 18 19 20 21 23 24 26 28 30 32 34 35 38 39 43 44 45 46 47 49 50"

resolve_seed_for_idx() {
    local env_seed="$1"
    local idx="$2"

    if [[ ! -v WORKING_SEEDS[$env_seed] ]]; then
        echo "No seed list for environment seed ${env_seed}" >&2
        exit 1
    fi

    local -a candidates=()
    # shellcheck disable=SC2206  # word splitting intentional to build array
    read -r -a candidates <<< "${WORKING_SEEDS[$env_seed]}"

    if ((${#candidates[@]} == 0)); then
        echo "Empty seed list for environment seed ${env_seed}" >&2
        exit 1
    fi

    if (( idx < 0 || idx >= ${#candidates[@]} )); then
        echo "Index ${idx} out of range for environment seed ${env_seed}" >&2
        exit 1
    fi

    echo "${candidates[$idx]}"
}

# ---------------Configs--------- 
CONFIG="config_options_base"
OPTION_TYPE="MaskedOptionLearner"
NAME_TAG="PPO_Move_Actions_Mask_input_$IDX" #"Distractor_MaxLen-20_Mask-l1_$IDX"
SEED=$IDX
EXP_PATH_LIST=()

for ENV_SEED in "${ENVIRONMENT_SEEDS[@]}"; do
    ACTUAL_SEED="$(resolve_seed_for_idx "$ENV_SEED" "$IDX")"
    EXP_PATH_LIST+=(
        "Runs/Train/MiniHack-Corridor-R2-v0_seed-${ENV_SEED}_view_size-9/PPO/${ACTUAL_SEED}_seed[${ACTUAL_SEED}]"
    )
done

INFO='{
    "max_option_len": 20,

    "max_num_options": 5,
    "n_neighbours": 100,
    "n_restarts": 300,
    "n_iteration": 200,

    "n_epochs": 500,
    "actor_lr": 5e-4,

    "reg_coef": 0.00,
    "masked_layers":["input"]
}' 

RUN_IND_LIST=(1 1 1) #1 1 1 1 1)
NUM_WORKERS=16
# ----------------------------------

# Invoke Python script with all arguments
python learn_options.py \
  --config            "$CONFIG" \
  --option_type       "$OPTION_TYPE" \
  --seed              "$SEED" \
  --name_tag          "$NAME_TAG" \
  --exp_path_lst      "${EXP_PATH_LIST[@]}" \
  --run_ind_lst       "${RUN_IND_LIST[@]}" \
  --num_workers       "$NUM_WORKERS" \
  --info              "$INFO"


echo "---- SLURM JOB STATS ----"
seff $SLURM_JOBID || sacct -j $SLURM_JOBID --format=JobID,ReqMem,MaxRSS,Elapsed,State

# Temporary line
python clean_storage.py Runs/Options/ --condition selected_options_5.t --target all_options.t --apply
