import os, time, argparse
import cv2 as cv
import numpy as np

from functools import partial
from pathlib import Path

# globals
frame_device = cv.cuda_GpuMat()


def clearImage(frame):

    channels = cv.split(frame)

    # Get the maximum value of each channel
    # and get the dark channel of each image
    # record the maximum value of each channel
    a_max_dst = [float("-inf")] * len(channels)
    for idx in range(len(channels)):
        a_max_dst[idx] = channels[idx].max()

    dark_image = cv.min(channels[0], cv.min(channels[1], channels[2]))

    # Gaussian filtering the dark channel
    dark_image = cv.GaussianBlur(dark_image, (25, 25), 0)

    image_t = (255.0 - 0.95 * dark_image) / 255.0
    image_t = cv.max(image_t, 0.5)

    # Calculate t(x) and get the clear image
    for idx in range(len(channels)):
        channels[idx] = (
            cv.max(
                cv.add(
                    cv.subtract(channels[idx].astype(np.float32), int(a_max_dst[idx]))
                    / image_t,
                    int(a_max_dst[idx]),
                ),
                0.0,
            )
            / int(a_max_dst[idx])
            * 255
        )
        channels[idx] = channels[idx].astype(np.uint8)

    return cv.merge(channels)


def ProcVid(proc_frame_func, vidPath):
    cap = cv.VideoCapture(vidPath)
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        return
    n_frames = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            n_frames += 1
            proc_frame_func(frame)
        else:
            break
    end = time.time()
    cap.release()
    return (end - start) * 1000 / n_frames, n_frames


def ProcFrameCuda(frame, size=(416, 416)):
    # frame_device.upload(frame)
    # change frame to frame_device below for gpu version
    frame_device_small = cv.resize(frame, dsize=size)
    fg_device = cv.cvtColor(frame_device_small, cv.COLOR_BGR2RGB)
    fg_device = clearImage(fg_device)
    # fg_host = fg_device.download()
    store_res = True
    if store_res:
        gpu_res.append(np.copy(fg_device))


def main():
    "Handles argument parsing and launches the correct function."
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_path", "-v", help="path to video file", type=str)
    parser.add_argument(
        "--cols", "-c", "--cols", help="model input columns", type=int, default=416
    )
    parser.add_argument("--rows", "-r", help="model input rows", type=int, default=416)
    args = parser.parse_args()

    # Run tests
    gpu_time_0, n_frames = ProcVid(
        partial(ProcFrameCuda, size=(args.cols, args.rows)), args.vid_path
    )
    print(f"GPU 0 (naive): {n_frames} frames, {gpu_time_0:.2f} ms/frame")


if __name__ == "__main__":
    gpu_res = []
    main()
