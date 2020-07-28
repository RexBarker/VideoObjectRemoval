#docker run --ipc=host --gpus all -ti -v"${PWD}/..:/inpainting" inpainting:latest bash
docker run --ipc=host --gpus all -ti \
	-v"${PWD}/../../data:/home/appuser/data" \
	-v"${PWD}/../../setup:/home/appuser/setup" \
	-v"${PWD}/../scripts:/home/appuser/scripts" \
	-v"${PWD}/../pretrained_models:/home/appuser/Deep-Flow/pretrained_models" \
	-p 48172:22 \
	--network detectinpaint \
	--name=inpaint inpainting:latest bash

