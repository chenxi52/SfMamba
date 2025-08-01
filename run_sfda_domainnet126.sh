########### train source ##############
data_root=/your_dataroot/
lr=3e-4

# config_file=configs/vmambav2v_tiny_224.yaml 
# model_path=utils/vssm1_tiny_0230s_ckpt_epoch_264.pth
# source_dir=outputs/vmamba_tiny

config_file=configs/vmambav2_small_224.yaml 
model_path=utils/vssm_small_0229_ckpt_epoch_222.pth 
source_dir=outputs/vmamba_small

python main_source.py --cfg $config_file --data_path ${data_root}/domainnet126/ --lr $lr \
		--output $source_dir --dataset domainnet126  --test_envs 1 2 3 \
        --pretrained $model_path --dg_aug \
        --batch_size 64 --test_batch_size 32 \
        TRAIN.WARMUP_EPOCHS 3 TRAIN.EPOCHS 30 &&
# python main_source.py --cfg $config_file --data_path ${data_root}/domainnet126/ --lr $lr \
# 		--output $source_dir --dataset domainnet126  --test_envs 0 2 3 \
#         --pretrained $model_path --dg_aug \
#         --batch_size 64 --test_batch_size 32 \
#         TRAIN.WARMUP_EPOCHS 3 TRAIN.EPOCHS 30&&
# python main_source.py --cfg $config_file --data_path ${data_root}/domainnet126/ --lr $lr \
# 		--output $source_dir --dataset domainnet126  --test_envs 0 1 3 \
#         --pretrained $model_path --dg_aug \
#         --batch_size 64 --test_batch_size 32 \
#         TRAIN.WARMUP_EPOCHS 3 TRAIN.EPOCHS 30&&
# python main_source.py --cfg $config_file --data_path ${data_root}/domainnet126/ --lr $lr \
# 		--output $source_dir --dataset domainnet126  --test_envs 0 1 2 \
#         --pretrained $model_path --dg_aug \
#         --batch_size 64 --test_batch_size 32 \
#         TRAIN.WARMUP_EPOCHS 3 TRAIN.EPOCHS 30

################ train target ##################
lr=3e-5
epoch=5
source_dir=outputs/vmamba_small
out_dir=${source_dir}_target

##CPRS
st_cas="20 21 10 03 31 23 12"  
for pair in $st_cas; do
    s=${pair:0:1} 
    t=${pair:1:1} 
    echo "s=$s, t=$t"
    python main_target.py --cfg $config_file \
        --data_path ${data_root}/domainnet126/ --lr $lr \
        --output $out_dir --dataset domainnet126 \
        --target_env $t --issave \
        --batch_size 32 --source_env $s --test_batch_size 16 \
        MODEL.SOURCE_DIR $source_dir TRAIN.EPOCHS $epoch \
        TRAIN.WARMUP_EPOCHS 1 
done
