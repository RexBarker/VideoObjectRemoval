export PUBLIC_IP=`curl ifconfig.io`

docker run --gpus all -it  --shm-size=8gb --env="DISPLAY" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume=$HOME/.torch/fvcore_cache:/tmp:rw  \
	-p 8889:8889/tcp -p 48171:22 \
	--name=detect_inpaintv1 detect_inpaint:v1
