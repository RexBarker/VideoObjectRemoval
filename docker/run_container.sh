export PUBLIC_IP=`curl ifconfig.io`

docker run --gpus all -it  --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume=$HOME/.torch/fvcore_cache:/tmp:rw  -p 8889:8889/tcp --name=detect_inpaint detect_inpaint:v0
