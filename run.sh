## Usage: bash run.sh <s2t_loss_ratio> <t2t_loss_ratio> <threshold_loss>
## s2t_loss_ratio: S2T loss fraction
## t2t_loss_ratio: T2T loss fraction
## threshold_loss: Threshold loss

s2t_loss_ratio=$1
t2t_loss_ratio=$2
threshold_loss=$3
BASE_MODEL="seamlessM4T_medium" # Options: seamlessM4T_medium, seamlessM4T_large, seamlessM4T_large_v2
LANG_CODE="<lang_code>" # Language code for the target language, See Seamless documentation for more details
PARENT_DIR="<path_to_parent_dir>"
EXPERIMENT_NAME="<experiment_name>"
EXPERIMENT_DIR="$PARENT_DIR/$EXPERIMENT_NAME"
DATASET_DIR="<path_ containing_manifest_files>"
GPUs="<gpu_id>"
mkdir -p $EXPERIMENT_DIR

## Logging
LOG_FILE="$EXPERIMENT_DIR/output.log"
if [ -f "$LOG_FILE" ]; then
    echo -e "\e[31mWarning: Log file already exists. Appending to the file.\e[0m"
fi
exec > >(tee -a "$LOG_FILE") 2>&1
export CUDA_VISIBLE_DEVICES=$GPUs
export CUBLAS_WORKSPACE_CONFIG=:4096:8


if [ $s2t_loss_ratio -le 0 ] && [ $t2t_loss_ratio -le 0 ]; then
    echo "Both s2t_loss_ratio and t2t_loss_ratio are less than or equal to zero. Exiting."
    exit 1
fi

warmup_steps=<warmup_steps>
max_epochs=<max_epochs>
eval_steps=<eval_steps>
batch_size=<batch_size>
log_steps=<log_steps>


# Finetuning
echo "Step 1: Finetuning"
torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--nnodes=1 \
--nproc-per-node=1 \
--no-python \
m4t_finetune \
--mode SPEECH_TO_PARA \
--train_dataset $DATASET_DIR/train_manifest.json \
--eval_dataset $DATASET_DIR/validation_manifest.json \
--learning_rate 5e-6 \
--warmup_steps $warmup_steps \
--max_epochs $max_epochs  \
--eval_steps $eval_steps \
--batch_size $batch_size \
--patience 5 \
--log_steps $log_steps \
--s2t_loss_ratio $s2t_loss_ratio \
--t2t_loss_ratio $t2t_loss_ratio \
--threshold_loss $threshold_loss \
--model_name $BASE_MODEL \
--save_model_to $EXPERIMENT_DIR/$EXPERIMENT_NAME.pt \
2>&1 | tee $EXPERIMENT_DIR/$EXPERIMENT_NAME.log
