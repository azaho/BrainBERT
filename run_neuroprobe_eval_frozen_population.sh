#!/bin/bash
#SBATCH --job-name=e_bb_lite          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=2    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH -t 1:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-72
#SBATCH --output logs/%A_%a.out # STDOUT
#SBATCH --error logs/%A_%a.err # STDERR
#SBATCH --open-mode=append  # Append to output files instead of overwriting
#SBATCH --requeue
#SBATCH -p mit_preemptable

export PYTHONUNBUFFERED=1
source .venv/bin/activate

export ROOT_DIR_BRAINTREEBANK=/orcd/data/fiete/001/zaho/braintreebank_laplacian_rereferenced_line_noise_removed/
echo "Running on $(hostname)"

# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
declare -a subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
declare -a trials=(1 2 0 4 0 1 0 1 0 1 0 1)

declare -a eval_names=(
    "frame_brightness"
    "global_flow"
    "local_flow"
    "face_num"
    "volume"
    "pitch"
    "delta_volume"
    "speech"
    "onset"
    "gpt2_surprisal"
    "word_length"
    "word_gap"
    "word_index"
    "word_head_pos"
    "word_part_speech"
)
eval_names=$(IFS=,; echo "${eval_names[*]}") # Join the eval names with commas to run them in sequence

declare -a splits_type=(
    "WithinSession"
    "CrossSession"
    "CrossSubject"
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


# Check if we're trying to evaluate subject 2 with DS_DM split (which is invalid)
if [[ "$SPLITS_TYPE" == "DS_DM" && "$SUBJECT" == "2" ]]; then
    echo "Cannot evaluate the cross subject split on subject 2; exiting"
    exit 0
fi


SAVE_DIR="eval_results_${SPLITS_TYPE}"

nvidia-smi

echo "Using python: $(which python)"
echo "Using python version: $(python --version)"

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, splits_type $SPLITS_TYPE, feature_type $FEATURE_TYPE, random_init $RANDOM_INIT"
echo "Command: python -u run_neuroprobe_eval_frozen_population.py --only_1second --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --split_type $SPLITS_TYPE --save_dir $SAVE_DIR --feature_type $FEATURE_TYPE --randomly_initialized_model $RANDOM_INIT"

# Add the -u flag to Python to force unbuffered output
python -u run_neuroprobe_eval_frozen_population.py --only_1second --eval_name $EVAL_NAME --subject_id $SUBJECT --trial_id $TRIAL --split_type $SPLITS_TYPE --save_dir $SAVE_DIR --feature_type $FEATURE_TYPE --randomly_initialized_model $RANDOM_INIT