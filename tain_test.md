# Latent Bridge Matching (LBM) - ICCV 2025 (Highlight)

## Training

```bash
    cd /root/shuqian/LBM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SLURM_NPROCS=1
export SLURM_NNODES=1
export SLURM_PROCID=0
export SLURM_JOB_ID=0

CUDA_VISIBLE_DEVICES=0 python examples/training/train_lbm_surface.py \
  --path_config examples/training/config/ar_surface.yaml
  
  CUDA_VISIBLE_DEVICES=0 python examples/training/train_lbm_surface.py \
  examples/training/config/ar_surface.yaml
```
### SD 15


```.bash
cd /root/shuqian/projects/LBM-main

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

python infer_ar_sd15.py \
  --config examples/training/config/ar_surface_sd15v2.yaml \
  --ckpt /root/storage/shuqian/LBM_out/AR_sd15_bs2_lpips10_321ep/last.ckpt \
  --input_dir /root/shuqian/dataset/AR_LBM_raw/test_A \
  --output_dir /root/storage/shuqian/LBM_out/AR_sd15_bs2_lpips10_321ep/test_pred_lpips10_step1 \
  --num_steps 1 \
  --size 256
  
  python infer_ar_sd15.py \
  --config examples/training/config/ar_surface_sd15v2.yaml \
  --ckpt /root/storage/shuqian/LBM_out/AR_sd15_bs2_lpips10_321ep/last.ckpt \
  --input_dir /root/shuqian/dataset/AR_LBM_raw/test_A \
  --output_dir /root/storage/shuqian/LBM_out/AR_sd15_bs2_lpips10_321ep/test_pred_lpips10_step2 \
  --num_steps 2 \
  --size 256
  
    python infer_ar_sd15.py \
  --config examples/training/config/ar_surface_sd15v2.yaml \
  --ckpt /root/storage/shuqian/LBM_out/AR_sd15_bs2_lpips10_321ep/last.ckpt \
  --input_dir /root/shuqian/dataset/AR_LBM_raw/test_A \
  --output_dir /root/storage/shuqian/LBM_out/AR_sd15_bs2_lpips10_321ep/test_pred_lpips10_step4 \
  --num_steps 4 \
  --size 256
  
  
  
  CUDA_VISIBLE_DEVICES=1 python examples/training/train_lbm_surface_sd15.py \
  --path_config examples/training/config/ar_surface_sd15v2.yaml
```
tmux new -s lbm-5-5
tmux attach -t lbm-5-5
tmux attach -t bi-lbm-6-18
exit

## 打包
原始数据解构
/root/shuqian/dataset/AR_LBM_raw/train_A
/root/shuqian/dataset/AR_LBM_raw/train_B
/root/shuqian/dataset/AR_LBM_raw/test_A
/root/shuqian/dataset/AR_LBM_raw/test_B

执行
```.bash
cd /root/shuqian/projects/LBM-main

mkdir -p /root/shuqian/dataset/AR_LBM_wds

rm -f /root/shuqian/dataset/AR_LBM_wds/ar_train-*.tar
rm -f /root/shuqian/dataset/AR_LBM_wds/ar_val-*.tar

python make_ar_lbm_webdataset.py \
  --sketch_dir /root/shuqian/dataset/AR_LBM_raw/train_A \
  --photo_dir /root/shuqian/dataset/AR_LBM_raw/train_B \
  --out_pattern /root/shuqian/dataset/AR_LBM_wds/ar_train-%06d.tar \
  --maxcount 100000

python make_ar_lbm_webdataset.py \
  --sketch_dir /root/shuqian/dataset/AR_LBM_raw/test_A 9+999999\
  --photo_dir /root/shuqian/dataset/AR_LBM_raw/test_B \
  --out_pattern /root/shuqian/dataset/AR_LBM_wds/ar_val-%06d.tar \
  --maxcount 100000
```
## 训练
```.bash
cd /root/shuqian/projects/Bi-LBM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SLURM_NPROCS=1
export SLURM_NNODES=1
export SLURM_PROCID=0
export SLURM_JOB_ID=0

CUDA_VISIBLE_DEVICES=0 python examples/training/train_lbm_surface_sd15.py \
  --path_config examples/training/config/ar_surface_bilbm_twobridge_v4.yaml
```
## 1.生成face parsing
### 生成训练集 parsing
```.bash
cd /root/shuqian/projects/LBM-main

python tools/generate_face_parsing.py \
  --input_dir /root/shuqian/dataset/AR_LBM_raw/train_B \
  --output_dir /root/shuqian/dataset/AR_LBM_raw/train_parse \
  --decp_root /root/shuqian/projects/DECP \
  --bisenet_ckpt /root/shuqian/projects/DECP/pretrained_models/face_parsing_bisenet.pth \
  --size 256 \
  --device cuda
```
### 生成测试集 paring
```.bash
python tools/generate_face_parsing.py \
  --input_dir /root/shuqian/dataset/AR_LBM_raw/test_B \
  --output_dir /root/shuqian/dataset/AR_LBM_raw/test_parse \
  --decp_root /root/shuqian/projects/DECP \
  --bisenet_ckpt /root/shuqian/projects/DECP/pretrained_models/face_parsing_bisenet.pth \
  --size 256 \
  --device cuda
```
## 2.打包数据集
```.bash
cd /root/shuqian/projects/LBM-main

python make_ar_lbm_webdataset.py \
  --sketch_dir /root/shuqian/dataset/AR_LBM_raw/train_A \
  --photo_dir /root/shuqian/dataset/AR_LBM_raw/train_B \
  --parse_dir /root/shuqian/dataset/AR_LBM_raw/train_parse \
  --out_pattern "/root/shuqian/dataset/AR_LBM_wds/ar_train-%06d.tar" \
  --maxcount 1000 \
  --require_parse
```

```.bash
python make_ar_lbm_webdataset.py \
  --sketch_dir /root/shuqian/dataset/AR_LBM_raw/test_A \
  --photo_dir /root/shuqian/dataset/AR_LBM_raw/test_B \
  --parse_dir /root/shuqian/dataset/AR_LBM_raw/test_parse \
  --out_pattern "/root/shuqian/dataset/AR_LBM_wds/ar_val-%06d.tar" \
  --maxcount 1000 \
  --require_parse
```

cp /root/storage/shuqian/LBM_out/ar-v4-5-14-id0.05-no-id-crop/last.ckpt \
   /root/storage/shuqian/LBM_out/ar-v5-parse-adapter-v1/last.ckpt

CUDA_VISIBLE_DEVICES=1 python tools/check_training_parse_visualization.py \
  --shards "/root/shuqian/dataset/AR_LBM_wds/ar_train-{000000..000000}.tar" \
  --out_dir debug_parse_check \
  --num_samples 8 \
  --simulate_mapper totensor_rescale \
  --run_adapter