# AUTHOR: LINYUN LIU

# INTRODUCTION: This program applies the black and white image algorithm to colorize video
# by spliting it into individual frames first. After colorizing each of the frames
# we put them back together according to a specific frame rate.
# * Audio track will be lost in the output video (will update in the future)

# The program is inspired by the YouTube Video: https://www.youtube.com/watch?v=oNjQpq8QuAo
# Richzhang is the main contributor: https://github.com/richzhang/colorization
# This program is an elegant combiniation of different algorithms
# * Please pre-install opencv-python, numpy packages

import os
import time
import numpy as np
import cv2

# Use absolute path if necessary
# Find the link to download the models in the README.md file
prototxt_path = '/models/colorization_deploy_v2.prototxt'
model_path = '/models/colorization_release_v2.caffemodel'
kernel_path = '/models/pts_in_hull.npy'

bw_video_path = '/PATH/TO/VIDEO/input.mp4' # Black and white video input path
work_place_path = "/PATH/TO/WORKPLACE" # Set a work place for the colorization process
colorized_video_path = '/PATH/TO/VIDEO/output.mp4' # Destination path for the colorized video

# video spliting function
def split_video(source):
    # To Find out how many frames the video have
    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame count: " + str(length))

    # Create a folder for frames
    newpath = work_place_path + '/Frames'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    vidcap = cv2.VideoCapture(source)
    success, image = vidcap.read()
    count = 0

    while success:
        # save frame as png file
        cv2.imwrite(work_place_path + '/Frames/frame%d.png' % count, image)
        success, image = vidcap.read()
        print(str(count + 1) + "/" + str(length))
        count += 1

# Colorization algorithm
def colorize_image(bw_image_path, out_image_path):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)

    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype='float32')]

    bw_image = cv2.imread(bw_image_path)
    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")

    cv2.imwrite(out_image_path, colorized)


# Frames combining function
def combine_frames():
    path = work_place_path + '/ColorizedFrames/'
    out = colorized_video_path

    frames = os.listdir(path)
    img = []
    length = len(frames)

    print("Gathering Frames...")
    count = 0
    for i in frames:
        img.append(path + 'frame' + str(count) + '.png')
        count = count + 1

    cap = cv2.VideoCapture(bw_video_path)
    FPS = float(cap.get(cv2.CAP_PROP_FPS))

    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(img[0])
    size = list(frame.shape)
    del size[2]
    size.reverse()
    video = cv2.VideoWriter(out, cv2_fourcc, FPS, size)

    print("writing file...")
    count = 0
    for i in range(len(img)):
        video.write(cv2.imread(img[i]))
        print(str(count + 1) + "/" + str(length))
        count = count + 1
    video.release()


    
### HERE TO START THE PROCESSES ###
# input is the source video
# Frames will be saved in Frames folder
# We need to split the video into frames first
print("Starting to split...")
time.sleep(3)
split_video(bw_video_path)
print("\033[92mVideo splitting completed!\033[0m\n")

# Preparing for colorizing
# Create a folder for all colorized frames
colorized_frames_folder_path = work_place_path + '/ColorizedFrames'
if not os.path.exists(colorized_frames_folder_path):
    os.makedirs(colorized_frames_folder_path)
# to get the frames from the Frames folder
frame_folder_path = work_place_path + '/Frames'
frame_list = os.listdir(frame_folder_path)
length = len(frame_list)
count = 0
print("Starting to colorize frames...")
time.sleep(3)
for i in frame_list:
    colorize_image(frame_folder_path + "/frame" + str(count) + ".png",
                   colorized_frames_folder_path + "/frame" + str(count) + ".png")
    print(str(count + 1) + " out of " + str(length) + " completed")
    count = count + 1
print("\033[92mColorizing completed!\033[0m\n")

# Now it's time to combine all the colorized frames
print("Starting to combine all the colorized frames...")
time.sleep(3)
combine_frames()
print("\033[92mCompleted!\033[0m\n")

# That's it, not too complecated, the result will not be the best, but at least decent.


