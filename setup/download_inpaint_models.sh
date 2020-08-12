# this script automatically downloads from a google drive location
# as originally posted by https://drive.google.com/drive/folders/1Nh6eJsue2IkP_bsN02SRPvWzkIi6cNbE

# you can use this script, or go to this link if you don't trust my script

DownloadScript=./_download_googledrive.sh
DownloadDir=../inpaint/pretrained_models

$DownloadScript 1rKr1HtqjJ5gBdOA8fTgJ99brWDIJi46v $DownloadDir/FlowNet2_checkpoint.pth.tar
$DownloadScript 1ZxyGeWk1d37QdZkx1d2aBseXSfYL4uuJ $DownloadDir/resnet101-5d3b4d8f.pth
$DownloadScript 1i0fZ37se14p7-MW5fxi6O4rv3VjHCDFd $DownloadDir/resnet101_movie.pth
$DownloadScript 1dZQjITK8bOWuS4yQeC_WVddqT_QkFXFO $DownloadDir/resnet50-19c8e357.pth
$DownloadScript 16hmQgpp_cPBzw5Dug9EnjVnwy7a6KF9M $DownloadDir/resnet50_stage1.pth
$DownloadScript 1jltdGzyZaJ1RGpeMf9Ns6ofyEJMsyboe $DownloadDir/imagenet_deepfill.pth
