if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <test_dataset> <root_dir> <train_dataset> <ckpt_path>"
    exit 1
fi

TEST_DATASET=$1
ROOT_DIR=$2
TRAIN_DATASET=$3
CKPT_PATH=$4

python3 hybridmqa/run/test.py \
    --dataset "$TEST_DATASET" \
    --root_dir "$ROOT_DIR" \
    --tr_dataset "$TRAIN_DATASET" \
    --ckpt "$CKPT_PATH" \
    --norm \
    --batch_size 2 \
    --num_work 2 \
    --pin_mem \
    --test_all \
    --rloss