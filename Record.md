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


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" \
 --operator "srx8" \
 --output_dir "sd2cn_srx8_1000_re/" \
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


CUDA_VISIBLE_DEVICES=0 proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_50_500s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_50/checkpoint-5000/controlnet --step 500

python -m pytorch_fid /NEW_EDS/JJ_Group/xutd/common_datasets/imagenet_512x512/train /NEW_EDS/JJ_Group/xutd/diffusion-inversion/results/imagenet/srx8/dps/recon

CUDA_VISIBLE_DEVICES=0 nohup proxychains python -u main_sd2cn.py --data ./imagenet_512 --out ./results/imagenet/srx8/dpscn_100_1000s --cnmodel /NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sd2cn_srx8_100/checkpoint-5000/controlnet --step 1000 --scale 1.2 &> imagenet_srx8_dpscn_100_1000s.out &

nohup

&> imagenet_srx8_dps_1000s_sd1.5.out

## DPS 1000s

|S-chain|-<>-127.0.0.1:1080-<><>-4.2.2.2:53-<><>-OK
|DNS-response| huggingface.co is 18.172.78.5
|S-chain|-<>-127.0.0.1:1080-<><>-18.172.78.5:443-<><>-OK
100%|██████████| 1000/1000 [05:35<00:00,  2.98it/s, distance=2.17]
[0/1000], avg mse: 0.011384
100%|██████████| 1000/1000 [05:35<00:00,  2.98it/s, distance=2.7]
[1/1000], avg mse: 0.023542
100%|██████████| 1000/1000 [05:35<00:00,  2.98it/s, distance=2.91]
[2/1000], avg mse: 0.019106
100%|██████████| 1000/1000 [05:35<00:00,  2.98it/s, distance=2.51]
[3/1000], avg mse: 0.015461
100%|██████████| 1000/1000 [05:35<00:00,  2.98it/s, distance=3]  
[4/1000], avg mse: 0.018846
100%|██████████| 1000/1000 [05:35<00:00,  2.98it/s, distance=2.81]
[5/1000], avg mse: 0.028385
100%|██████████| 1000/1000 [05:34<00:00,  2.99it/s, distance=3.19]
[6/1000], avg mse: 0.031479
100%|██████████| 1000/1000 [05:34<00:00,  2.99it/s, distance=2.85]
[7/1000], avg mse: 0.028788
100%|██████████| 1000/1000 [05:34<00:00,  2.99it/s, distance=2.91]
[8/1000], avg mse: 0.027391
100%|██████████| 1000/1000 [05:34<00:00,  2.99it/s, distance=2.42]
[9/1000], avg mse: 0.025287
100%|██████████| 1000/1000 [05:34<00:00,  2.99it/s, distance=3.37]
[10/1000], avg mse: 0.027416
100%|██████████| 1000/1000 [05:34<00:00,  2.99it/s, distance=2.84]



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

CUDA_VISIBLE_DEVICES=1 proxychains python stablesr.py

CUDA_VISIBLE_DEVICES=1 proxychains python main_sd3.py


CUDA_VISIBLE_DEVICES=1,2,3,6 nohup accelerate launch --num_processes 4 --main_process_port 29502 train_stablesr.py \
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
 --gradient_accumulation_steps=3 \
 --num_train_epochs=10000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=500 &> sdsr_srx8_full.out &
