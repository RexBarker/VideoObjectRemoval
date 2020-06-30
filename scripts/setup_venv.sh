# to be run by docker build 
# activate virtualenv
. /home/appuser/venv/bin/activate

# install into virtualenv
apt update -y; 
sudo apt install -y python3 python3-pip libglib2.0-0 libsm6 libxext6 libxrender-dev
pip3 --disable-pip-version-check install addict==2.2.1 certifi==2019.6.16 cffi==1.12.3 chardet==3.0.4 cvbase==0.5.5 Cython==0.29.12 idna==2.8 mkl-fft mkl-random numpy==1.16.4 olefile==0.46 opencv-python==4.1.0.25 Pillow==6.2.0 protobuf==3.8.0 pycparser==2.19 PyYAML==5.1.1 requests==2.22.0 scipy==1.2.1 six==1.12.0 tensorboardX==1.8 terminaltables==3.1.0 torch==0.4.0 torchvision==0.2.1 tqdm==4.32.1 urllib3==1.25.3
pip3 --disable-pip-version-check install mmcv==0.2.10

# install kernel for jupyter notebook to this venv
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "venv"


