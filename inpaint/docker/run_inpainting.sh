#docker run --ipc=host --gpus all -ti -v"${PWD}/..:/inpainting" inpainting:latest bash
docker run --ipc=host --gpus all -ti \
	-v"${PWD}/../../data:/home/appuser/data" \
	-v"${PWD}/../scripts:/home/appuser/scripts" \
	-v"${PWD}/../pretrained_models:/home/appuser/Deep-Flow/pretrained_models" \
	--name=inpaint inpainting:latest bash

