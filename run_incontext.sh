#dataset=$1
device=$1

#[ -z "${dataset}" ] && dataset="ogbn-arxiv"
[ -z "${device}" ] && device=0

CUDA_VISIBLE_DEVICES=$device \
python main.py \
    --run_entity \
    --device 0 \
    --seed 0 \
    --mask_rate 0.5 \
    --dropout 0.0 \
    --drop_edge_rate 0.0 \
    --num_layers 2 \
    --lr 2e-5 \
    --weight_decay 0.001 \
    --lr_f 1e-2 \
    --cut_off 128 \
    --batch_size 60 \
    --eval_batch_size 256 \
    --num_epochs 1 \
    --num_roots 20 \
    --length 20 \
    --gnn_type gat \
    --lp_epochs 5000 \
    --eval_steps 10000 \
    --hidden_size 768 \
    --lm_type microsoft/deberta-base \
    --datasets_name papers100M \
    --eval_datasets_name cora \
    --process_mode TA \
    --lam 0.1 \
    --load_checkpoint \
    --incontext_eval \
    --eval_num_label 2 \
    --eval_num_support 3 \
    --eval_num_query 3 