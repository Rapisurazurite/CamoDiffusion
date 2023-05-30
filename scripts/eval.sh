MULTI_CARD=true
RANDOM_RUN_ID=$(shuf -i 100000-999999 -n 1)
RESULTS_FOLDER="/tmp/CamoDiffusion_debug/${RANDOM_RUN_ID}"


SAMPLING_STEPS=(10)
BATCH_SIZE=32
TARGET_DATASET="CAMO COD10K NC4K"

LAUNCH_COMMAND="python"
if ${MULTI_CARD}; then
    RANDOM_PORT=$(shuf -i 20000-29999 -n 1)
    LAUNCH_COMMAND="accelerate launch --num_machine=1 --multi_gpu --num_processes=2 --gpu_ids=0,1 --mixed_precision=no --main_process_port ${RANDOM_PORT}"
fi

CONFIG_FILE="config/camoDiffusion_384x384.yaml" # Change to your config file
CHECKPOINT="CHECKPOINT_PATH" # Change to your checkpoint


# echo Config file and checkpoint
echo -e "\033[32m [Config file: ${CONFIG_FILE}] \033[0m"
echo -e "\033[32m [Checkpoint: ${CHECKPOINT}] \033[0m"

for i in "${SAMPLING_STEPS[@]}";do
    # Check if the results folder exists, if so, delete it
    if [ -d "${RESULTS_FOLDER}" ]; then
        echo -e "\033[31m [Results folder ${RESULTS_FOLDER} exists, deleting...] \033[0m"
        rm -rf ${RESULTS_FOLDER}
    else
        echo -e "\033[32m [Results folder ${RESULTS_FOLDER} does not exist, creating...] \033[0m"
        mkdir -p ${RESULTS_FOLDER}
    fi
    echo -e "\033[32m Sampling ${i} steps \033[0m"
    ${LAUNCH_COMMAND} sample.py \
      --config=${CONFIG_FILE} \
      --results_folder=${RESULTS_FOLDER} \
      --checkpoint=${CHECKPOINT} \
      --batch_size=${BATCH_SIZE} \
      --num_sample_steps=${i} \
      --target_dataset ${TARGET_DATASET} \
      --time_ensemble
done
