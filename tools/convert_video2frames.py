import argparse
import cv2
import os
import numpy as np
from math import log10, ceil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, default=None,
                        help="input video file (.avi, .mp4, .mkv, mov)")
    parser.add_argument('--rotate_right', action='store_true', help="Rotate image by 90 deg clockwise")
    parser.add_argument('--rotate_left', action='store_true', help="Rotate image by 90 deg anticlockwise")
    parser.add_argument('--image_type', type=str, default='png', help="output frame file type (def=png)")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="name of output directory (default = base of input file name")

    args = parser.parse_args()

    return args

def video_to_frames(inputfile,outputdir,imagetype='png'):

    if not os.path.exists(outputdir):
        dout = '.'
        for din in outputdir.split('/'):
            dout = dout + '/' + din
            if not os.path.exists(dout):
                os.mkdir(dout)

    cap = cv2.VideoCapture(inputfile)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    padlength = ceil(log10(length))

    n = 0
    while True:
        ret, frame = cap.read() 

        if not ret: break

        if args.rotate_left:
            frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif args.rotate_right:
            frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        
        fname = str(n).rjust(padlength,'0') + '.' + imagetype
        cv2.imwrite(os.path.join(outputdir,fname),frame) 

        n += 1

    return n  # number of frames processed


if __name__ == '__main__':
    args = parse_args()

    assert os.path.exists(args.input_file), f"Could not find input file = {args.input_file}"
    inputfile = args.input_file

    currdir = os.path.abspath(os.curdir)

    if args.output_dir is not None:
        outputdir = args.output_dir
    else:
        outputdir = os.path.basename(inputdir).split('.')[0]
        outputdir = os.path.join(currdir,outputdir + "_frames") 

    n = video_to_frames(inputfile,outputdir,imagetype=args.image_type) 

    print(f"\nCompleted successfully, processed {n} frames")
