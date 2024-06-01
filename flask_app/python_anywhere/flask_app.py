import os
from flask import Flask, flash, request, redirect, render_template
from create_gif import gif
from pose_detection import pose_detection
from trim_video import trim_video
from classification import classification
import warnings
import time
import cv2
import sys


# Hide warnings
warnings.simplefilter('ignore')

UPLOAD_FOLDER = '/home/YuuS/mysite/static/upload'
ALLOWED_EXTENSIONS = set(['mp4'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "sample123"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    t_start = time.time()
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Error: File not found', 'failed')
            return redirect(request.url)
        file = request.files['file']
        print('Input file name: ', file.filename)
        if file.filename == '':

            flash('Error: File not found', 'failed')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Error: File format must be mp4', 'failed')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file.save(os.path.join(UPLOAD_FOLDER, "video.mp4"))

            # Check if video length meets the requirement.
            video_capture = cv2.VideoCapture("/home/YuuS/mysite/static/upload/video.mp4")
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            video_seconds = total_frames // fps
            if video_seconds < 10:
                flash('Error: Video is too short.', 'failed')
                return redirect(request.url)
            elif video_seconds > 1800:
                flash('Error: Video is too long. Please trim your video.', 'failed')
                return redirect(request.url)

            trim_video()
            filepath = os.path.join(UPLOAD_FOLDER, "video_trimmed.mp4")
            hand = request.form.get('radio_hand')
            if hand == 'right_handed':
                is_lefty=False
            elif hand == 'left_handed':
                is_lefty=True
            gif(filepath, "/home/YuuS/mysite/static/upload/video.gif", lefty=is_lefty)

            # Run pose-detection
            mode = request.form.get('radio_mode')

            print("Running pose-detection.")
            pose_detection(mode)

            # Run classification model
            pred = classification(mode)
            print("pred_result_video: ", pred[4])

            t_end = time.time()
            elapsed_time = t_end - t_start
            print('Elapsed time: {:.2f} minutes'.format(elapsed_time / 60))

            if mode == 'advanced':
                # Advanced mode
                return render_template('result.html', msg_f=pred[0], msg_n=pred[1], msg_d=pred[2], msg_m=pred[3],
                                       result_video=pred[4], msg_sub1=pred[5], msg_sub2=pred[6], msg_sub3=pred[7])
            else:
                # Quick mode
                return render_template('result_quick.html', msg_f=pred[0], msg_n=pred[1], msg_d=pred[2], msg_m=pred[3],
                                       result_video=pred[4])

    return render_template('index.html')


# Run app
if __name__ == '__main__':
    sys.exit(app.run())
