import os
import cv2
import numpy as np
import argparse
import multiprocessing
import shutil
import cv2

from grayscale_to_rgb_motion import convert_to_rgb

data_dir= '/l/users/ganzorig.batnasan/data/asl-citizen/ASL_Citizen/videos'
dest_dir = '/l/users/ganzorig.batnasan/data/asl-citizen/ASL_Citizen/videos_gray_to_rgb'

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Convert a video to RGB")

    # Add the video_path argument
    #parser.add_argument("data_dir", type=str, default = ' ',help="Path to the video file")

    # Parse the command-line arguments
    #args = parser.parse_args()

    webm_videos = [(os.path.join(data_dir,video),os.path.join(dest_dir,video)) for video in os.listdir(data_dir) if video.endswith('.mp4')]

    # Create a pool of worker processes
    pool = multiprocessing.Pool()

    # Convert the videos in parallel
    pool.map(convert_to_rgb, webm_videos)


    # Close the pool
    pool.close()
    pool.join()
