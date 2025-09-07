#!/bin/bash
#SBATCH --job-name=e_bb_lite          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=2    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=10GB
#SBATCH --exclude=dgx001,dgx002
#SBATCH --mem=128G
#SBATCH -t 1:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-1368 #1-456 # 267-302
#SBATCH --output logs/%A_%a.out # STDOUT
#SBATCH --error logs/%A_%a.err # STDERR
#SBATCH -p use-everything

export PYTHONUNBUFFERED=1
source .venv/bin/activate

export ROOT_DIR_BRAINTREEBANK=/om2/user/zaho/braintreebank_laplacian_rereferenced_line_noise_removed/
echo "Running on $(hostname)"


# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
declare -a subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
declare -a trials=(1 2 0 4 0 1 0 1 0 1 0 1)

declare -a eval_names=(
    "frame_brightness"
    "global_flow"
    "local_flow"
    "global_flow_angle"
    "local_flow_angle" 
    "face_num"
    "volume"
    "pitch"
    "delta_volume"
    "delta_pitch"
    "speech"
    "onset"
    "gpt2_surprisal"
    "word_length"
    "word_gap"
    "word_index"
    "word_head_pos"
    "word_part_speech"
    "speaker"
)
declare -a splits_type=(
    "SS_SM"
    "SS_DM"
    "DS_DM"
)
declare -a feature_type=(
    "keepall"
)
declare -a random_init=(
    "0"
    "1"
)

# Calculate indices for this task
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
FEATURE_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#feature_type[@]} ))
RANDOM_INIT_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#feature_type[@]} % ${#random_init[@]} ))
SPLITS_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#feature_type[@]} / ${#random_init[@]} % ${#splits_type[@]} ))   

# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
SPLITS_TYPE=${splits_type[$SPLITS_TYPE_IDX]}
FEATURE_TYPE=${feature_type[$FEATURE_TYPE_IDX]}
RANDOM_INIT=${random_init[$RANDOM_INIT_IDX]}

SAVE_DIR="eval_results_${SPLITS_TYPE}"

nvidia-smi

echo "Using python: $(which python)"
echo "Using python version: $(python --version)"

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, splits_type $SPLITS_TYPE, feature_type $FEATURE_TYPE, random_init $RANDOM_INIT"
echo "Command: python -u run_neuroprobe_eval_frozen_population.py --only_1second --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --split_type $SPLITS_TYPE --save_dir $SAVE_DIR --feature_type $FEATURE_TYPE --randomly_initialized_model $RANDOM_INIT"

# Add the -u flag to Python to force unbuffered output
python -u run_neuroprobe_eval_frozen_population.py --only_1second --eval_name $EVAL_NAME --subject_id $SUBJECT --trial_id $TRIAL --split_type $SPLITS_TYPE --save_dir $SAVE_DIR --feature_type $FEATURE_TYPE --randomly_initialized_model $RANDOM_INIT