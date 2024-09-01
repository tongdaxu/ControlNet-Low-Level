## ControlNet

/home/xutd/.cache/huggingface/accelerate/default_config.yaml

CUDA_VISIBLE_DEVICES=4,5 nohup accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "edge" \
 --output_dir "sd2cn_edge_1000/" \
 --train_data ./imagenet512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=8 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &

vim /home/xutd/.cache/huggingface/accelerate/default_config.yaml


CUDA_VISIBLE_DEVICES=1 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_1000 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000/checkpoint-5000/controlnet &> imagenet_srx8_dpscn_1000.out &

CUDA_VISIBLE_DEVICES=6 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_100 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet &> imagenet_srx8_dpscn_100.out &

CUDA_VISIBLE_DEVICES=2 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_50 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_50/checkpoint-5000/controlnet &> imagenet_srx8_dpscn_50.out &

CUDA_VISIBLE_DEVICES=5 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_25 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_25/checkpoint-4000/controlnet &> imagenet_srx8_dpscn_25.out &

CUDA_VISIBLE_DEVICES=3 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_25 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_25/checkpoint-4000/controlnet &> imagenet_srx8_dpscn_25.out &

CUDA_VISIBLE_DEVICES=7 nohup proxychains python -u main_sd2.py --data ./ffhq_512 --out ./results/ffhq/srx8/dps --scale 6.4 &> ffhq_srx8_dps.out &

CUDA_VISIBLE_DEVICES=7 proxychains python -u main_sd2.py --data ./ffhq_512 --out ./results/ffhq/srx8/dps --scale 6.4

CUDA_VISIBLE_DEVICES=1 proxychains python -u main_sd2cn.py --data ./gen_512 --out ./gen_512 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/sd2cn_srx8_1000/checkpoint-5000/controlnet --disablecn


CUDA_VISIBLE_DEVICES=7 nohup proxychains python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/psld --scale 4.8 --mode psld &> imagenet_srx8_psld.out &

CUDA_VISIBLE_DEVICES=6 nohup proxychains python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/stsl --scale 4.8 --mode stsl &> imagenet_srx8_stsl.out &

## SRx8

CUDA_VISIBLE_DEVICES=1 nohup proxychains python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/dps --mode dps --step 1000 &> imagenet_srx8_dps.out &

CUDA_VISIBLE_DEVICES=2 nohup proxychains python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/dps_500s --mode dps --step 500 &> imagenet_srx8_dps_500s.out &


CUDA_VISIBLE_DEVICES=2 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_1000 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000/checkpoint-5000/controlnet --step 250 --scale 2.4 &> imagenet_srx8_dpscn_1000.out &

CUDA_VISIBLE_DEVICES=3 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_100 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 250 --scale 3.6 &> imagenet_srx8_dpscn_100.out &

CUDA_VISIBLE_DEVICES=4 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_50 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_50/checkpoint-5000/controlnet --step 250 &> imagenet_srx8_dpscn_50.out &

CUDA_VISIBLE_DEVICES=5 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_25 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_25/checkpoint-4000/controlnet --step 250 &> imagenet_srx8_dpscn_25.out &

CUDA_VISIBLE_DEVICES=6 nohup proxychains python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/pt --mode pt --step 1000 &> imagenet_srx8_pt.out &

CUDA_VISIBLE_DEVICES=7 nohup proxychains python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/stsl --mode stsl --step 500 &> imagenet_srx8_stsl.out &

CUDA_VISIBLE_DEVICES=7 proxychains python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/stsl --mode stsl --step 500

CUDA_VISIBLE_DEVICES=3 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_100_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 500 --scale 2.4 &> imagenet_srx8_dpscn_100_500s.out &

CUDA_VISIBLE_DEVICES=4 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_1000_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000/checkpoint-5000/controlnet --step 500 &> imagenet_srx8_dpscn_1000_500s.out &


CUDA_VISIBLE_DEVICES=5 proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/edge/dpscn_1000 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_edge_1000/checkpoint-8000/controlnet --step 250 --operator edge

CUDA_VISIBLE_DEVICES=4 proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_1000_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000/checkpoint-5000/controlnet --step 500

CUDA_VISIBLE_DEVICES=4 proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_1000_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000/checkpoint-5000/controlnet --step 500


CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com nohup accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "gdb" \
 --output_dir "sd2cn_gdb_1000/" \
 --train_data ./imagenet512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &


CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com nohup accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "gdb" \
 --output_dir "sd2cn_gdb_100/" \
 --train_data ./imagenet512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &> train_gdb_100.out &


CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com nohup accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "gdb" \
 --output_dir "sd2cn_gdb_100/" \
 --train_data ./gen512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &> train_gdb_gen.out &

CUDA_VISIBLE_DEVICES=0 proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_50_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_50/checkpoint-5000/controlnet --step 500

python -m pytorch_fid /NEW_EDS/JJ_Group/xutd/common_datasets/imagenet_512x512/train /NEW_EDS/JJ_Group/xutd/diffusion-inversion/results/imagenet/srx8/dps/recon

CUDA_VISIBLE_DEVICES=0 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_100_1000s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 1000 --scale 1.2 &> imagenet_srx8_dpscn_100_1000s.out &

nohup

&> imagenet_srx8_dps_1000s_sd1.5.out

## DPS 1000s

CUDA_VISIBLE_DEVICES=1 proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/tmp1 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 250

CUDA_VISIBLE_DEVICES=3 proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_100_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 500 --scale 4.8

CUDA_VISIBLE_DEVICES=3 nohup proxychains python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_100_500s --step 500 --scale 4.8 &> imagenet_srx8_dpscn_100_500s.out &


CUDA_VISIBLE_DEVICES=6 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_100_500s_4.8 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 500 --scale 4.8 &> imagenet_srx8_dpscn_100_500s_4.8.out &


CUDA_VISIBLE_DEVICES=7 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_1000_500s_4.8 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000_re/checkpoint-10000/controlnet --step 500 --scale 4.8 &> imagenet_srx8_dpscn_1000_500s_4.8.out &

CUDA_VISIBLE_DEVICES=2 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_full_500s_4.8 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_full/checkpoint-10000/controlnet --step 500 --scale 4.8 &> imagenet_srx8_dpscn_full_500s_4.8.out &

CUDA_VISIBLE_DEVICES=6 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_25_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_25/checkpoint-4000/controlnet --step 500 --scale 4.8 &> imagenet_srx8_dpscn_25_500s.out &


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "srx8" \
 --output_dir "sd2cn_srx8_gen1000/" \
 --train_data ./gen512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &> cn_gen1000_train.out &

CUDA_VISIBLE_DEVICES=2 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_50_500s_4.8 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_50/checkpoint-5000/controlnet --step 500 --scale 4.8 &> imagenet_srx8_dpscn_50_500s_4.8.out &


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch train_controlnet_dps.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "srx8" \
 --output_dir "sd2cn_srx8_full_dps/" \
 --train_data ./imagenet512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=16 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &

CUDA_VISIBLE_DEVICES=6 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_1000_1000s_1.2 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000_re/checkpoint-10000/controlnet --step 1000 --scale 1.2 &> imagenet_srx8_dpscn_1000_1000s_1.2.out &

CUDA_VISIBLE_DEVICES=7 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_1000_1000s_2.4 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000_re/checkpoint-10000/controlnet --step 1000 --scale 2.4 &> imagenet_srx8_dpscn_1000_1000s_2.4.out &

CUDA_VISIBLE_DEVICES=5 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_100_1000s_2.4 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 1000 --scale 2.4 &> imagenet_srx8_dpscn_100_1000s_2.4.out &


CUDA_VISIBLE_DEVICES=7 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_100_1000s_4.8 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 1000 --scale 4.8 &> imagenet_srx8_dpscn_100_1000s_4.8.out &

CUDA_VISIBLE_DEVICES=3 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_full_half_500s_4.8 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/sd2cn_srx8_full/checkpoint-5000/controlnet --step 500 --scale 4.8 &> imagenet_srx8_dpscn_full_half_500s_4.8.out &

CUDA_VISIBLE_DEVICES=4 proxychains nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_full_quad_500s_4.8 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/sd2cn_srx8_full/checkpoint-2500/controlnet --step 500 --scale 4.8 &> imagenet_srx8_dpscn_full_quad_500s_4.8.out &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --main_process_port 29501 train_stablesr.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "srx8" \
 --output_dir "sdsr_srx8_full/" \
 --train_data ./imagenet512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=6 \
 --gradient_accumulation_steps=3 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &


CUDA_VISIBLE_DEVICES=4 proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/tmp --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000_re/checkpoint-10000/controlnet --step 500 --scale 1.2 


CUDA_VISIBLE_DEVICES=4 proxychains python -u main_sd2.py --data /NEW_EDS/JJ_Group/xutd/StableSR/example --out ./results/imagenet/srx8/examples --step 500 --scale 4.8 

CUDA_VISIBLE_DEVICES=1 proxychains python -u main_sd2.py --data /NEW_EDS/JJ_Group/xutd/StableSR/example --out ./results/imagenet/srx8/tmp2 --scale 1.2 --step 999 &> ffhq_srx8_dps.out &

CUDA_VISIBLE_DEVICES=1 python stablesr.py

CUDA_VISIBLE_DEVICES=1 proxychains python main_sd3.py


CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --num_processes 4 --main_process_port 29502 train_stablesr.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "srx8" \
 --output_dir "sdsr_srx8_full/" \
 --train_data ./imagenet512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=5e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &> sdsr_srx8_full.out &


CUDA_VISIBLE_DEVICES=5 accelerate launch --num_processes 1 --main_process_port 29503 train_stablesr.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "srx8" \
 --output_dir "sdsr_srx8_full/" \
 --train_data ./imagenet512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=5e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=1


CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com  nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/stsl --scale 4.8 --mode stsl --step 50 &

CUDA_VISIBLE_DEVICES=7 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/psld_500s --scale 4.8 --mode psld


CUDA_VISIBLE_DEVICES=7 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/psld_500s --scale 4.8 --mode psld --step 500

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/pt_500s --scale 4.8 --mode pt --step 500 &

CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/psld_500s --scale 4.8 --mode psld --step 500 &



CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python -u main_sdsr.py --data ./tmp --out ./results/imagenet/realsrx4/stablesr --scale 4.8 --mode dps

CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com python -u stablesr.py

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet_align/srx8/dps_1000s --scale 1.0  --mode dps --step 999 &> dps_ddim_1000.out &

CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/pt_500s_le100 --scale 4.8 --mode pt --step 500 &> pt_500_le100.out &

CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet_align/srx8/dpscn_100_1000s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 999 --scale 1.0 &> dpscn_100_ddim_1000.out &


CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sdsr.py --data ./imagenet_512 --out ./results/imagenet/realsrx4/dps_bicubicx4_0.2 --scale 0.2 --mode dps &> sdsr_srx4_dps_0.2.out &

CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sdsr.py --data ./imagenet_512 --out ./results/imagenet/realsrx4/dps_bicubicx4_0.2 --scale 0.2 --mode dps &> sdsr_srx4_dps_0.2.out &


CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sdsr.py --data ./imagenet_512 --out ./results/imagenet/realsrx4/stablesr_bicubicx4 --scale 0.0 --mode dps &> sdsr_srx4_nodps.out &

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sdsr.py --data /NEW_EDS/JJ_Group/xutd/BSRGAN/testsets/RealSRSet --out ./results/imagenet/realsrx4/dps_0.2_real --scale 0.2 --mode dps &> sdsr_real_0.2.out &

CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sdsr.py --data /NEW_EDS/JJ_Group/xutd/BSRGAN/testsets/RealSRSet --out ./results/imagenet/realsrx4/stablesr_real --scale 0.0 --mode dps &> sdsr_real_nodps.out &


CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sdsr.py --data ./imagenet_512 --out ./results/imagenet/realsrx4/stablesr_dps_0.02 --scale 0.02 --mode dps &> sdsr_dps_0.02.out &

CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com python -u main_sdsr.py --data ./imagenet_512 --out ./results/imagenet/realsrx4/stablesr_dps_0.02_k --scale 0.02 --mode dps

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python -u main_sdsr.py --data ./imagenet_512 --out ./results/imagenet/realsrx4/stablesr_dps_0.02_bf --scale 0.02 --mode dps &> sdsr_dps_0.02_bf.out &

CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com python -u main_sdsr.py --data ./imagenet_512 --out ./results/imagenet/realsrx4/stablesr_dps_0.02_bf_k --scale 0.02 --mode dps &> sdsr_dps_0.02_bf_k.out &


CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com python -u main_sdsr.py --data ./imagenet_512 --out ./results/imagenet/realsrx4/stablesr_dps_0.0_bf --scale 0.0 --mode dps &> sdsr_dps_0.0_bf.out &



CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/dsg_500s_0.08 --scale 0.08 --mode dsg --step 500 &> srx8_dsg_0.08.out &

CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dsgcn_100_500s_0.08 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 500 --scale 0.08 --mode dsg &> srx8_dsgcn_0.08.out &

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dsgcn_1000_500s_0.01 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000_re/checkpoint-10000/controlnet --step 500 --scale 0.01 --mode dsg &> srx8_dsgcn_1000_0.01.out &

CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dsgcn_1000_500s_0.02 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000_re/checkpoint-10000/controlnet --step 500 --scale 0.02 --mode dsg &> srx8_dsgcn_1000_0.02.out &

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dsgcn_1000_500s_0.04 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000_re/checkpoint-10000/controlnet --step 500 --scale 0.04 --mode dsg &> srx8_dsgcn_1000_0.04.out &



CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_1000_gen_500s_4.8 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_gen1000/checkpoint-10000/controlnet --step 500 --scale 4.8 --mode dps &> srx8_dpscn_1000_gen_4.8.out &



CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/cn_100_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 500 --scale 0.0 --mode cn &> srx8_cn_100.out &

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/cn_100_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 500 --scale 0.0 --mode cn &> srx8_cn_100.out &


CUDA_VISIBLE_DEVICES=6 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/cn_1000_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000_re/checkpoint-10000/controlnet --step 500 --scale 0.0 --mode cn &> srx8_cn_1000.out &



CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com python -u main_sd15.py --out ./results/lsun_bedroom/layout/dps --scale 0.15 --step 500

CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com python main_lcm.py 


CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com python -u main_sd15.py --out ./results/lsun_bedroom/layout/dpscm --scale 0.02 --step 500 --mode dpscm &> lsun_dpscm.out &


CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python -u mean_score.py --data ./imagenet_512 --out ./tmp --scale 4.8 --mode dps --step 500 

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python -u mean_score.py --data ./imagenet_512 --out ./tmp --scale 1.2 --mode dps --step 500 


CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd15.py --out ./results/lsun_bedroom/layout/dpscm --scale 9.6 --step 500 --mode dpscm &> roomlayout_dpscm.out &

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python -u main_sd15.py --out ./results/lsun_bedroom/layout/dps --scale 9.6 --step 500 --mode dps &> roomlayout_dps.out &

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python -u main_sd15.py --out ./results/lsun_bedroom/layout/lgd --scale 9.6 --step 500 --mode lgd &> roomlayout_lgd.out &


CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/dps_500s_1.2 --scale 1.2 --mode dps --step 500 &> srx8_dps_1.2.out &


CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/dps_500s_0.3 --scale 0.3 --mode dps --step 500 &> srx8_dps_0.3.out &

CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/dps_500s_0.05 --scale 0.05 --mode dps --step 500 &> srx8_dps_0.05.out &

CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/dps_500s_0.01 --scale 0.01 --mode dps --step 500 &> srx8_dps_0.01.out &



CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/gdb/dps_500s --mode dps --step 500 --scale 0.6 --operator gdb &> imagenet_gdb_dps_500s.out &

CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/gdb/dpscn_1000_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_gdb_1000/checkpoint-8500/controlnet --step 500 --scale 0.6 --mode dps --operator gdb &> imagenet_gdb_dpscn_1000_500s.out &

CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/gdb/dpscn_gen_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_gdb_gen/checkpoint-6500/controlnet --step 500 --scale 0.6 --mode dps --operator gdb &> imagenet_gdb_dpscn_gen_500s.out &


CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/gdb/dpscn_100_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_gdb_100/checkpoint-6500/controlnet --step 500 --scale 0.6 --mode dps --operator gdb &> imagenet_gdb_dpscn_100_500s.out &


CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/gdb/dsg_500s --scale 0.08 --mode dsg --step 500 --operator gdb &> gdb_dsg_0.08.out &

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/gdb/dsgcn_100_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_gdb_100/checkpoint-6500/controlnet --step 500 --scale 0.02 --mode dsg --operator gdb &> gdb_dsgcn_100_0.02.out &

CUDA_VISIBLE_DEVICES=6 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/gdb/dsgcn_1000_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_gdb_1000/checkpoint-8500/controlnet --step 500 --scale 0.02 --mode dsg --operator gdb &> gdb_dsgcn_1000_0.02.out &

CUDA_VISIBLE_DEVICES=7 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/gdb/dsgcn_gen_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_gdb_gen/checkpoint-6500/controlnet --step 500 --scale 0.02 --mode dsg --operator gdb &> gdb_dsgcn_gen_0.02.out &

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/cn_100_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 500 --scale 0.0 --mode cn &> srx8_cn_100.out &

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dsgcn_gen_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_gen1000/checkpoint-10000/controlnet --step 500 --scale 0.02 --mode dsg --operator srx8 &> srx8_dsgcn_gen_0.02.out &


CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/pr/dps_500s --scale 4.8 --mode dps --step 500 --operator pr &> pr_dps_4.8.out &


CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/nlb/dps_500s --scale 4.8 --mode dps --step 5 --operator nlb 


CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com nohup accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "pr" \
 --output_dir "sd2cn_pr_1000/" \
 --train_data ./imagenet512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &> train_pr_1000.out &


CUDA_VISIBLE_DEVICES=2,3,4,7 HF_ENDPOINT=https://hf-mirror.com nohup accelerate launch --main_process_port 29501 train_controlnet.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "nlb" \
 --output_dir "sd2cn_nlb_100/" \
 --train_data ./imagenet512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=8 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &> train_nlb_100.out &

CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com nohup accelerate launch --main_process_port 29501 train_controlnet.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "nlb" \
 --output_dir "sd2cn_nlb_gen/" \
 --train_data ./gen512.txt \
 --val_data ./imagenet_512 \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=8 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &> train_nlb_gen.out &

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/max_500s_0.08 --scale 0.08 --mode max --step 500 &> srx8_max_0.08.out

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/max_500s_4.8 --scale 4.8 --mode max --step 500 &> srx8_max_4.8.out &

CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/maxcn_100_500s_3.6 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 500 --scale 3.6 --mode max &> srx8_maxcn_100_3.6.out &

CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/maxcn_1000_500s_2.4 --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_1000_re/checkpoint-10000/controlnet --step 500 --scale 2.4 --mode max &> srx8_maxcn_1000_2.4.out &

CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/gdb/max_500s --step 500 --scale 0.6 --operator gdb --mode max &> imagenet_gdb_max_500s.out &

CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/gdb/maxcn_100_500s --step 500 --scale 0.6 --operator gdb --mode max --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_gdb_100/checkpoint-6500/controlnet &> imagenet_gdb_maxcn_100_500s.out &

CUDA_VISIBLE_DEVICES=7 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/gdb/maxcn_1000_500s --step 500 --scale 0.6 --operator gdb --mode max --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_gdb_1000/checkpoint-8500/controlnet &> imagenet_gdb_maxcn_1000_500s.out &

CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/psld --scale 4.8 --mode psld &> imagenet_srx8_psld.out &

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/gdb/psld --mode psld --step 500 --scale 0.6 --operator gdb &> imagenet_gdb_psld.out &

CUDA_VISIBLE_DEVICES=6 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/rsp --mode rsp --step 500 --scale 4.8 --operator srx8 &> imagenet_srx8_rsp.out &

CUDA_VISIBLE_DEVICES=7 HF_ENDPOINT=https://hf-mirror.com nohup python -u main_sd2.py --data ./imagenet_512 --out ./results/imagenet/srx8/fdm --mode fdm --step 500 --scale 4.8 --operator srx8 &> imagenet_srx8_fdm.out &


 &> imagenet_gdb_psld.out &

