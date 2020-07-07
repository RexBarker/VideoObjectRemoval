# Grant docker access to host X server to show images
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' detectron2`
