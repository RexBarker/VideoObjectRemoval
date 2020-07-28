# create docker network for interprocess communication
if [ -z `docker network ls -f name=detectinpaint -q` ]; then
 docker network create -d bridge detectinpaint 
 echo "created detectinpaint network"
else
 echo "detectinpaint network already exists"
fi

