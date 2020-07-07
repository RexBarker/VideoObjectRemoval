# run the detector for a sample image: ../data/input.jpg
# result is stored in ../data/outputs

DATA=/home/appuser/data
BASE=/home/appuser/detectron2_repo

wget http://images.cocodataset.org/val2017/000000439715.jpg -O ../../data/input.jpg

docker exec -ti detect \
	/usr/bin/python3 $BASE/demo/demo.py \
	--config-file $BASE/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
       	--input $DATA/input.jpg \
	--output $DATA/outputs/ \
	--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
