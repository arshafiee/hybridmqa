if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <root_dir>"
    exit 1
fi

ROOT_DIR=$1

python3 hybridmqa/run/train.py \
    --dataset VCMesh \
    --root_dir "$ROOT_DIR" \
    --norm \
    --ckpt_step 3 \
    --batch_size 1 \
    --lr 0.001 \
    --num_epochs 3 \
    --num_work 2 \
    --pin_mem \
    --shuffle_seed 3 \
    --ang_aug \
    --flip_aug \
    --rloss