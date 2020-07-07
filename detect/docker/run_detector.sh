export PUBLIC_IP=`curl ifconfig.io`

docker run --gpus all -it  --shm-size=8gb --env="DISPLAY" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume=$HOME/.torch/fvcore_cache:/tmp:rw  \
	--volume="${PWD}/../../data:/home/appuser/data" \
	--volume="${PWD}/../scripts:/home/appuser/scripts" \
	-p 8889:8889/tcp -p 48171:22 \
	--name=detect detector:v0

