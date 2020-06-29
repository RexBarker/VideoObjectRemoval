FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cython
RUN pip install --user torch==1.5+cu101 torchvision==0.6+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# install Deep-Flow
RUN git clone https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting.git Deep-Flow 

RUN sudo apt-get update && sudo apt-get install -y vim
RUN sudo apt-get update && sudo apt-get install -y ffmpeg 
RUN pip install --user -e detectron2_repo
RUN pip install jupyter

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

RUN sudo apt-get update && sudo apt-get install \
  -y --no-install-recommends python3 python3-virtualenv

# create virtual environment for DeepFlow
ENV VIRTUAL_ENV=/home/appuser/.venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /home/appuser/

RUN echo ". /home/appuser/.venv/bin/activate" > /home/appuser/setup_venv.sh \
 && echo "apt update -y; sudo apt install -y python3 python3-pip libglib2.0-0 libsm6 libxext6 libxrender-dev" >> /home/appuser/setup_venv.sh \
 && echo "pip3 --disable-pip-version-check install addict==2.2.1 certifi==2019.6.16 cffi==1.12.3 chardet==3.0.4 cvbase==0.5.5 Cython==0.29.12 idna==2.8 mkl-fft mkl-random numpy==1.16.4 olefile==0.46 opencv-python==4.1.0.25 Pillow==6.2.0 protobuf==3.8.0 pycparser==2.19 PyYAML==5.1.1 requests==2.22.0 scipy==1.2.1 six==1.12.0 tensorboardX==1.8 terminaltables==3.1.0 torch==0.4.0 torchvision==0.2.1 tqdm==4.32.1 urllib3==1.25.3" \
 >>  /home/appuser/setup_venv.sh \
 && echo "pip3 --disable-pip-version-check install mmcv==0.2.10" >> /home/appuser/setup_venv.sh \
 && chmod +x /home/appuser/setup_venv.sh

RUN sudo /home/appuser/setup_venv.sh


# working location
WORKDIR /home/appuser/
#WORKDIR /home/appuser/detectron2_repo


#ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser"]

# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
