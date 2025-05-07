if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <root_dir>"
    exit 1
fi

ROOT_DIR=$1

python3 hybridmqa/run/train.py \
    --dataset YN2023 \
    --root_dir "$ROOT_DIR" \
    --norm \
    --ckpt_step 15 \
    --batch_size 8 \
    --lr 0.0001 \
    --num_epochs 15 \
    --num_work 2 \
    --pin_mem \
    --shuffle_seed 1 \
    --kfold_seed 7 \
    --ang_aug \
    --flip_aug \
    --rloss