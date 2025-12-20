#!/bin/bash
data_root=/your_dataroot/
sub_dir=outputs/vmamba_small
config_file=configs/vmambav2_small_224.yaml 
model_path=utils/vssm_small_0229_ckpt_epoch_222.pth 

### inference SFDA target ####
for s in 0; do
    for i in 1; do
        if [ $i -eq $s ]; then
            continue
        fi
    echo "s=$s, t=${i}"
    python inference.py --cfg $config_file \
        --data_path ${data_root}/office-home/ \
        --output outputs/infer --dataset office-home --target_env $i \
        --source_env $s --test_batch_size 32  SEED 1 \
        MODEL.TARGET_PATH 'outputs/seed1'
    done
done