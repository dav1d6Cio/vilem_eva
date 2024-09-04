python -m torch.distributed.launch --nproc_per_node=1 --use_env Retrieval.py \
--config ./configs/finetune_config/coco.yaml \
--model models.vilem.VILEM --beta 0.2 \
--output_dir output/flick_finetune/vilem \
--checkpoint /group/30042/uasonchen/ALBEF/output/vilem_eva_coeff1e-2/checkpoint_00.pth
