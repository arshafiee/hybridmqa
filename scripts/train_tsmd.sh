if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <root_dir>"
    exit 1
fi

ROOT_DIR=$1

python3 hybridmqa/run/train.py \
    --dataset TSMD \
    --root_dir "$ROOT_DIR" \
    --norm \
    --ckpt_step 10 \
    --batch_size 4 \
    --lr 0.0001 \
    --num_epochs 20 \
    --num_work 2 \
    --pin_mem \
    --shuffle_seed 0 \
    --kfold_seed 7 \
    --ang_aug \
    --flip_aug \
    --rloss