# run the demo flamingo model using docker
docker exec -ti inpaint \
	/usr/bin/python3 /home/appuser/Deep-Flow/tools/video_inpaint.py \
	--frame_dir /home/appuser/data/flamingo/origin \
	--MASK_ROOT /home/appuser/data/flamingo/masks \
	--img_size 512 832 \
	--FlowNet2  --DFC --ResNet101 --Propagation
