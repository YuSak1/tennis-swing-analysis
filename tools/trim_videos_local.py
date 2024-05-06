import cv2
import math

# Output video length in seconds
trim_length = 10 

start_seconds = 5
stop_seconds = trim_length + start_seconds

# video_path = '../videos/Djokovic/'
# input_file = 'Video_v3/Djokovic1_v3.mp4'
# output_file = 'feature_extraction_add/d_1_'

video_path = '../videos/Test_videos/'
input_file = 'xxx.mp4'
output_file = 'test_short_videos/a_'

print("Input: ", video_path + input_file)

video_capture = cv2.VideoCapture(video_path + input_file)

fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

i=0
while math.ceil(fps*(stop_seconds)) <= total_frames:
    i += 1
    print(i)

    start_frame_index = math.ceil(fps * start_seconds)
    stop_frame_index = math.ceil(fps * stop_seconds)
    # Fix index if exceeds the video length
    if start_frame_index < 0:
        start_frame_index = 0
    if stop_frame_index >= total_frames:
        stop_frame_index = total_frames - 1

    # Capture the index
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    frame_index = start_frame_index

    img_arr = []
    while frame_index <= stop_frame_index:
        _, img = video_capture.read()
        img_arr.append(img)
        frame_index += 1

    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc("H", "2", "6", "4")
    video = None
    # Output file
    video_file_name = video_path + output_file + str(i) + '.mp4'

    for img in img_arr:
        if video is None:
            h, w, _ = img.shape
            video = cv2.VideoWriter(video_file_name, fourcc, fps, (w, h))
        video.write(img)
    video.release()

    # Increment seconds
    start_seconds += trim_length
    stop_seconds += trim_length
