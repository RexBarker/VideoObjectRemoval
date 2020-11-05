# Setup
---

### Docker Setup
(based on the [NVIDIA Docker installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker))

Here, the NVIDIA Docker installation for Ubuntu is described – specifically Ubuntu 18.04 LTS and 20.04 LTS.  At the time of this report, NVIDIA Docker platform is only supported on Linux.

**Remove existing older Docker Installations:**

If there an existing Docker installation, this should upgraded as necessary.  This project basis was constructed with Docker v19.03. <br>
`$ sudo apt-get remove docker docker-engine docker.io containerd runc`

**Install latest Docker engine:**

The following script can be used to install docker all repositories as required:
```
$ curl -fsSL https://get.docker.com -o get-docker.sh
$ sudo sh get-docker.sh
$ sudo sudo systemctl start docker && sudo systemctl enable docker
```

Add user name to docker run group: <br>
`$ sudo usermod -aG docker your-user`

**Install NVIDIA Docker:**

These installation instructions are based on the NVIDIA Docker documentation [6].  First, ensure that the appropriate NVIDIA driver is installed on the host system.  This can be tested with the following command.  This should produce a listing of the current GPU state, along with the current version of the driver:
`$ nvidia-smi`

If there is an existing earlier version of the NVIDIA Docker system (<=1.0), this must be first uninstalled: <br>
`$ docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f`

`$ sudo apt-get purge nvidia-docker`

Set the distribution package RPMs: 
```
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

**Install the nvidia-docker2 package:**
```
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
$ sudo systemctl restart docker
```

Test the NVIDIA Docker installation by executing nvidia-smi from within a container: <br>
`$ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

---
# Project Setup

Follow the instructions from the YouTube video for the specific project setup:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=YvSXwaDaxGA
" target="_blank"><img src="http://img.youtube.com/vi/YvSXwaDaxGA/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>

