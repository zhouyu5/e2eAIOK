#!/bin/bash

# step 1: build docker
# docker build docker/ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -f docker/Dockerfile.habana -t gaudi2_finetune:v1
# step 2: download data
# bash scripts/download_data.sh 
# mkdir -p model_save processed_data config && chmod -R a+w *
# step 3: preprocess data and run training
# bash scripts/gaudi_docker_run_all.sh
# step 4 (optional): debug
# sudo docker exec -it "gaudi_train" bash

sudo docker run \
    --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice --net=host \
    --name="gaudi_train" \
    --privileged \
    -v $(pwd):/workspace \
    -w /workspace/  \
    --device=/dev/dri \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=$no_proxy \
    -itd nathanzz2/gaudi2_finetune:v1


sudo docker exec "gaudi_train" bash -c "bash scripts/data_prepare_from_json_args.sh"
sudo docker exec "gaudi_train" bash -c "bash scripts/train_from_json_args.sh"