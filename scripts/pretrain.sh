python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py \
--config ./configs/pretrain_config/vilem.yaml \
--output_dir output/test \
--model models.vilem.VILEM --beta 0.2