python -m torch.distributed.launch --nproc_per_node=1 --use_env Retrieval.py \
--config ./configs/finetune_config/flickr.yaml \
--output_dir output/flick_finetune/vilem \
--checkpoint output/pretrain/vilem \
--model models.vilem.VILEM