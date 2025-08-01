############# train source ##############
data_root=/your_dataroot/
lr=3e-4

# config_file=configs/vmambav2v_tiny_224.yaml 
# model_path=utils/vssm1_tiny_0230s_ckpt_epoch_264.pth
# sub_dir=outputs/vmamba_tiny

sub_dir=outputs/vmamba_small
config_file=configs/vmambav2_small_224.yaml 
model_path=utils/vssm_small_0229_ckpt_epoch_222.pth 

python main_source.py --cfg $config_file \
        --data_path ${data_root}/office/ --lr $lr --pretrained $model_path \
		--output $sub_dir --dataset office  --test_envs 1 2 \
        --neck bn_relu --dg_aug \
        --batch_size 64 --test_batch_size 32 \
        TRAIN.WARMUP_EPOCHS 3 TRAIN.EPOCHS 100 &&

# python main_source.py --cfg $config_file \
#         --data_path ${data_root}/office/ --lr $lr --pretrained $model_path \
# 		--output $sub_dir --dataset office  --test_envs 0 2 \
#         --neck bn_relu --dg_aug \
#         --batch_size 64 --test_batch_size 32  \
#         TRAIN.WARMUP_EPOCHS 3 TRAIN.EPOCHS 100 &&

# python main_source.py --cfg $config_file \
#         --data_path ${data_root}/office/ --lr $lr --pretrained $model_path \
# 		--output $sub_dir --dataset office --test_envs 0 1 \
#         --neck bn_relu --dg_aug \
#         --batch_size 64 --test_batch_size 32 \
#         TRAIN.WARMUP_EPOCHS 3 TRAIN.EPOCHS 100 

# ################ train SFDA target ##################
lr=5e-5 
epoch=15
source_dir=outputs/vmamba_small
out_dir=${source_dir}_target

for s in 0; do
    for t in 0 1 2; do
        if [ "$s" -eq "$t" ]; then
            continue
        fi
    echo "s=$s, t=$t"
    python main_target.py --cfg $config_file \
        --data_path ${data_root}/office/ --lr $lr \
        --output $out_dir --dataset office --target_env ${t}  \
        --batch_size 32 --source_env $s --test_batch_size 16 --issave \
        MODEL.SOURCE_DIR $source_dir TRAIN.EPOCHS $epoch \
        TRAIN.WARMUP_EPOCHS 1 
    done
done


