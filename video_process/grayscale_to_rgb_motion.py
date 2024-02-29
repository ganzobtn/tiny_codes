import cv2
import numpy as np
import argparse
def convert_to_rgb(video_path):
    # Read the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total_frames:',total_frames)
    # Create a VideoWriter object to save the new RGB video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output_rgb_video.mp4', fourcc, fps, (width, height))

    # Read and process each frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Stack each 3 grayscale frames into one frame of an RGB video
        if frame_count % 3 == 0:
            stacked_frame = np.zeros((height, width, 3), dtype=np.uint8)
            stacked_frame[:, :, 0] = gray_frame
        elif frame_count % 3 == 1:
            stacked_frame[:, :, 1] = gray_frame
        elif frame_count % 3 == 2:
            stacked_frame[:, :, 2] = gray_frame
            out.write(stacked_frame)

        frame_count += 1

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print("RGB video saved successfully.")

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Convert a video to RGB")

    # Add the video_path argument
    parser.add_argument("video_path", type=str, help="Path to the video file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with the provided video_path
    convert_to_rgb(args.video_path)