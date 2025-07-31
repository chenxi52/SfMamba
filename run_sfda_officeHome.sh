########### train source ##############
data_root=/your_dataroot/
lr=3e-4

# config_file=configs/vmambav2v_tiny_224.yaml 
# model_path=utils/vssm1_tiny_0230s_ckpt_epoch_264.pth
# sub_dir=outputs/vmamba_tiny

sub_dir=outputs/vmamba_small
config_file=configs/vmambav2_small_224.yaml 
model_path=utils/vssm_small_0229_ckpt_epoch_222.pth 

python main_source.py --cfg $config_file \
     --data_path ${data_root}/office-home/ --lr $lr \
		--output $sub_dir --dataset office-home  --test_envs 1 2 3  \
        --pretrained $model_path --dg_aug --neck bn_relu \
        --batch_size 64 --test_batch_size 32  \
        TRAIN.WARMUP_EPOCHS 3 &&
python main_source.ppy --cfg $config_file \
        --data_path ${data_root}/office-home/ --lr $lr \
		--output $sub_dir --dataset office-home  --test_envs 0 2 3 \
        --pretrained $model_path --dg_aug --neck bn_relu  \
        --batch_size 64 --test_batch_size 32  \
        TRAIN.WARMUP_EPOCHS 3 &&
python main_source.py --cfg $config_file \
        --data_path ${data_root}/office-home/ --lr $lr \
		--output $sub_dir --dataset office-home  --test_envs 0 1 3 \
        --pretrained $model_path --dg_aug --neck bn_relu \
        --batch_size 64 --test_batch_size 32 \
        TRAIN.WARMUP_EPOCHS 3 &&
python main_source.py --cfg $config_file \
        --data_path ${data_root}/office-home/ --lr $lr \
		--output $sub_dir --dataset office-home  --test_envs 0 1 2 \
        --pretrained $model_path --dg_aug --neck bn_relu \
        --batch_size 64 --test_batch_size 32  \
        TRAIN.WARMUP_EPOCHS 3 

################ train SFDA target ##################
lr=3e-5
epoch=15
source_dir=outputs/vmamba_small
out_dir=${source_dir}_target

for s in 0 1 2 3; do
    for i in 0 1 2 3; do
        if [ $i -eq $s ]; then
            continue
        fi
    echo "s=$s, t=${i}"
    python main_target.py --cfg $config_file \
        --data_path ${data_root}/office-home/ --lr $lr --issave \
        --output $out_dir --dataset office-home --target_env $i \
        --batch_size 32 --source_env $s --test_batch_size 16 \
        MODEL.SOURCE_DIR $source_dir TRAIN.EPOCHS $epoch \
        TRAIN.WARMUP_EPOCHS 1 
    done
done
