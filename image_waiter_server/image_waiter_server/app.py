import os
import logging
import requests
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory
import tensorflow as tf
import numpy as np
import math
import uuid

logging.basicConfig(level=logging.INFO)
from image_waiter_server.image_processor import ImageProcessor

DEBUG = True

if DEBUG:
    WORK_FOLDER = '/media/roman/Other/celebA'
else:
    WORK_FOLDER = '/usr/work_dir'
CELEB_RAW_IMG_FOLDER = os.path.join(WORK_FOLDER, 'raw/raw')
ALIGNED_CELEB_IMG_FOLDER = os.path.join(WORK_FOLDER, 'aligned_sized')
ALIGNED_USER_IMG_FOLDER = os.path.join(WORK_FOLDER, 'tmp')
UPLOAD_FOLDER = os.path.join(WORK_FOLDER, 'users_images')
PRETRAINED_MODEL = os.path.join(WORK_FOLDER, 'pretrained_model')

app = Flask(__name__)
img_processor = ImageProcessor(
    aligned_img_folder=ALIGNED_CELEB_IMG_FOLDER,
    aligned_usr_img_folder=ALIGNED_USER_IMG_FOLDER,
    aligned_img_size=160,
    pretrained_model=PRETRAINED_MODEL)

@app.route('/image/<filename>')
def send_image(filename):
    return send_from_directory(CELEB_RAW_IMG_FOLDER, filename)

@app.route('/usr_image/<filename>')
def send_usr_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html')
        local_filename = uuid.uuid4().hex
        img_filepath = os.path.join(UPLOAD_FOLDER, local_filename)
        file.save(img_filepath)
        img_count = int(request.form.get("img_count"))

        best_matches_img_names = img_processor.get_best_matches_imgs(img_filepath, img_count)
        return render_template('index.html', usr_image_name=local_filename, image_names=best_matches_img_names)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

# how to run:
# from faces:
# CUDA_VISIBLE_DEVICES="" python image_waiter_server/app.py