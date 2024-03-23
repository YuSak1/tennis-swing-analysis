import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from create_gif import gif
from pose_detection import pose_detection
from trim_video import trim_video
import warnings

import cv2
from keras.models import load_model
from PIL import Image
import keras
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Hide warnings
warnings.simplefilter('ignore')

UPLOAD_FOLDER = './static/upload'
ALLOWED_EXTENSIONS = set(['mp4'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "sample123"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Error: File not found', 'failed')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':

            flash('Error: File not found', 'failed')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Error: File format must be mp4', 'failed')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # flash('Processing... this may take a few minutes.', 'success')
            file.save(os.path.join(UPLOAD_FOLDER, "video.mp4"))

            # Check if video length meets the requirement.
            video_capture = cv2.VideoCapture("./static/upload/video.mp4")
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            video_seconds = total_frames // fps
            if video_seconds < 10:
                flash('Error: Video is too short.', 'failed')
                return redirect(request.url)
            elif video_seconds > 60:
                flash('Error: Video is too long. Please trim your video.', 'failed')
                return redirect(request.url)

            trim_video()
            filepath = os.path.join(UPLOAD_FOLDER, "video_trimmed.mp4")
            gif(filepath, "static/upload/video.gif", lefty=False)

            # Run pose-detection
            print("Running pose-detection.")
            pose_detection()

            # model = load_model('model_weight/man_woman_cnn_v3.h5', compile=False)
            return render_template('result.html')

    return render_template('index.html')


# Run app
if __name__ == '__main__':
    app.run()
