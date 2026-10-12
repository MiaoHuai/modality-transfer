python test.py \
    --dataroot datasets/WHU-City/opt2sar \
    --name WHU-City_opt2sar_pix2pix \
    --model pix2pix \
    --num_test 700 \
    --gpu_ids 7 \
    --load_size 256 \
    --crop_size 256 \
    --direction BtoA