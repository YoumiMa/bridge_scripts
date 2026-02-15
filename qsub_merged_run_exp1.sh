cat $PE_HOSTFILE
echo $NHOSTS
echo "Number of slots: ${NSLOTS}"
# export MASTER_ADDR=$(cat $PE_HOSTFILE | head -1 | cut -d ' ' -f 1)
# export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
export MASTER_ADDR=$(head -n1 $PE_HOSTFILE | awk '{print $1}')
export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")


echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

NODE_TYPE="h100"
export NUM_GPU_PER_NODE=$(nvidia-smi -L | wc -l)

NUM_NODES=$(wc -l < "$PE_HOSTFILE")
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}

# 元のホストファイルを読んで、スロット数を48で割ってGPU数に変換
while read -r line; do
  hostname=$(echo "$line" | awk '{print $1}')
  slots=$(echo "$line" | awk '{print $2}')
  gpus=$((slots / 48))
  echo "${hostname} slots=${gpus}"
done < "$PE_HOSTFILE" > "$HOSTFILE_NAME"


echo $DATA_PARALLEL_SIZE

DATASET_DIR="/gs/bs/tga-okazaki/ma/data/smbcgic_merged_processed"

TRAIN_DATA_PATH=""

# smbc data
# DATASET_DIR配下の全てのサブディレクトリを追加
for FILE in "${DATASET_DIR}"/*; do
    if [[ "$FILE" == *.idx ]]; then
            BASENAME=$(basename "$FILE")
            
            # Remove _text_document.idx suffix
            NAME="${BASENAME%_text_document.idx}"
            # echo "Found dataset: $NAME"
            
            # Add to blended dataset path with weight 1
            TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1 ${DATASET_DIR}/${NAME}_text_document"
        fi
done

echo "TRAIN_DATA_PATH=$TRAIN_DATA_PATH"

# job name
MODEL_NAME=tokyotech-llm/Llama-3.1-Swallow-8B-v0.5
JOB_NAME="Llama-3.1-Swallow-8b-bridge-merged-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu"

export WORLD_SIZE=$NUM_GPUS
echo "world size is: ${WORLD_SIZE}"
echo "hostfile: $HOSTFILE_NAME"
cd /home/9/uj02089/fs_act-x/nemo-container/workspace/

CONTAINER_IMAGE="/gs/fs/tga-ma-act-x/nemo-container"
CONFIG_FILE="bridge_scripts/conf/llama31_merged_pretrain_exp1.yaml"

# run
mpirun -np $WORLD_SIZE \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x NCCL_P2P_LEVEL=NVL \
  -x PATH \
  -bind-to none \
  apptainer run --nv \
  --env MASTER_ADDR=$MASTER_ADDR \
  --env MASTER_PORT=$MASTER_PORT \
  -w -f -B /gs -B /apps -B /home -B /gs/fs/tga-okazaki/ma:/root ${CONTAINER_IMAGE} \
  python bridge_scripts/pretraining_llama3.py \
  --use-mpi \
  --config-file $CONFIG_FILE \
  --hf-path $MODEL_NAME \
  --data-path $TRAIN_DATA_PATH \
  --wandb-exp-name $JOB_NAME
