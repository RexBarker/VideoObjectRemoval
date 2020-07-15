import argparse
from glob import glob
import cv2
import os
import numpy as np
import subprocess as sp
import ffmpeg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, default=None,
                        help="input directory of frames (assuming numeric ordering)")

    parser.add_argument('--mask_dir', type=str, required=False, default=None,
                        help="(optional) input directory of masks (assuming numeric ordering)")

    parser.add_argument('--rotate_right', action='store_true', help="Rotate image by 90 deg clockwise")
    parser.add_argument('--rotate_left', action='store_true', help="Rotate image by 90 deg anticlockwise")
    parser.add_argument('--fps', type=int, default=25, help="frames per second encoding speed (default=25 fps)")
    parser.add_argument('--output_file', type=str, default=None,
                        help="name of output mp4 file (default = input directory name")

    args = parser.parse_args()

    return args


def createVideoClip_Cmd(clip, ouputfile, fps, size=[256, 256]):

    vf = clip.shape[0]
    command = ['ffmpeg',
               '-y',  # overwrite output file if it exists
               '-f', 'rawvideo',
               '-s', '%dx%d' % (size[1], size[0]),  # '256x256', # size of one frame
               '-pix_fmt', 'rgb24',
               '-r', '25',  # frames per second
               '-an',  # Tells FFMPEG not to expect any audio
               '-i', '-',  # The input comes from a pipe
               '-vcodec', 'libx264',
               '-b:v', '1500k',
               '-vframes', str(vf),  # 5*25
               '-s', '%dx%d' % (size[1], size[0]),  # '256x256', # size of one frame
               outputfile]
    # sfolder+'/'+name
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
    out, err = pipe.communicate(clip.tostring())
    pipe.wait()
    pipe.terminate()
    print(err)


def createVideoClip(clip, outputfile, fps, size=[256, 256]):

    vf = clip.shape[0]
     
    args = [#'ffmpeg',
               '-y',  # overwrite output file if it exists
               '-f', 'rawvideo',
               '-s', '%dx%d' % (size[1], size[0]),  # '256x256', # size of one frame
               '-pix_fmt', 'rgb24',
               '-r', str(fps),  # frames per second
               '-an',  # Tells FFMPEG not to expect any audio
               '-i', '-',  # The input comes from a pipe
               '-vcodec', 'libx264',
               '-b:v', '1500k',
               '-vframes', str(vf),  # 5*25
               '-s', '%dx%d' % (size[1], size[0]),  # '256x256', # size of one frame
               outputfile]
    
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(size[1], size[0]))
        .output(outputfile, pix_fmt='yuv420p', format='mp4', video_bitrate='1500k', r=str(fps), s='{}x{}'.format(size[1], size[0]))
        .overwrite_output()
    )
    
    command = ffmpeg.compile(process, overwrite_output=True)

    #command = ffmpeg.get_args(args, overwrite_output=True)
    # sfolder+'/'+name
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
    out, err = pipe.communicate(clip.tostring())
    pipe.wait()
    pipe.terminate()
    print(err)



if __name__ == '__main__':
    args = parse_args()

    out_frames = []

    assert os.path.exists(args.input_dir), f"Could not find input directory = {args.input_dir}"
    inputdir = args.input_dir

    imgfiles = []
    for ftype in ("*.jpg", "*.png"):
        imgfiles = sorted(glob(os.path.join(inputdir,ftype)))
        if imgfiles: break

    assert imgfiles, f"Could not find any suitable *.jpg or *.png files in {inputdir}" 

    # DAN, you left off here!
    if arg.mask_dir is not None:
        assert os.path.exists(args.mask_dir), f"Mask directory specified, but could not be found = {args.mask_dir}"

    fps = args.fps
    currdir = os.path.abspath(os.curdir)

    if args.output_file is not None:
        video_name = args.output_file
    else:
        video_name = os.path.basename(inputdir)
    if not video_name.endswith(".mp4"): video_name = video_name + ".mp4"

    for imgfile in imgfiles:
        print(imgfile)
        out_frame = cv2.imread(imgfile)

        if args.rotate_left:
            out_frame = cv2.rotate(out_frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif args.rotate_right:
            out_frame = cv2.rotate(out_frame,cv2.ROTATE_90_CLOCKWISE)

        shape = out_frame.shape
        out_frames.append(out_frame[:, :, ::-1])

    final_clip = np.stack(out_frames)

    outputfile = os.path.join(currdir,video_name)

    #createVideoClip(final_clip, outputfile, [shape[0], shape[1]])
    createVideoClip_Cmd(final_clip, outputfile, fps, [shape[0], shape[1]])
    print(f"\nVideo output file:{outputfile}")
    print("\nCompleted successfully")
