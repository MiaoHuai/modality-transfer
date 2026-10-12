python train.py \
    --dataroot datasets/WHU-City/opt2sar \
    --name WHU-City_opt2sar_pix2pix_512 \
    --model pix2pix \
    --direction BtoA \
    --gpu_ids 0,1 \
    --batch_size 16 \
    --load_size 600 \
    --crop_size 512
