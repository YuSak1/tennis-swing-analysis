import cv2
import math


def trim_video():
    # Trim first xx seconds
    start_seconds = 0
    stop_seconds = 10

    video_capture = cv2.VideoCapture("/home/YuuS/mysite/static/upload/video.mp4")
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
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

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = None
    # Output file
    video_file_name = "/home/YuuS/mysite/static/upload/video_trimmed.mp4"
    for img in img_arr:
        if video is None:
            h, w, _ = img.shape
            video = cv2.VideoWriter(video_file_name, fourcc, fps, (w, h))
        video.write(img)
    video.release()

    return None
