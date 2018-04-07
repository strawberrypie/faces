import os
import logging
import requests
import cv2
import facenet
from flask import Flask, render_template, request, url_for, send_from_directory
import tensorflow as tf
import numpy as np
import math

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

UPLOAD_FOLDER = 'users_images/raw'
ALIGNED_IMG_FOLDER = 'users_images/aligned'
MODEL_PATH = '../pretrained_model'
IMG_SIZE = 160

class ImgProcessor(object):
	def __init__(self, trained_model_path=MODEL_PATH):
		self._trained_model_path = trained_model_path

	def get_embeddding(self, img_path):
		embeddings_arr = None
	    with tf.Graph().as_default():
	        with tf.Session() as sess:
	            facenet.load_model(self._trained_model_path)
	            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
	            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
	            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
	            embedding_size = embeddings.get_shape()[1]

	            images = facenet.load_data([img_path], False, False, IMG_SIZE)
	            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            	embeddings_arr = sess.run(embeddings, feed_dict=feed_dict)
        return embeddings_arr

img_processor = ImgProcessor()

# logic - for now in this file
def align_image(img_filepath):
	_, img_name = os.path.split(user_img_path)
	aligned_img_path = os.path.join(ALIGNED_IMG_FOLDER, img_name)

    face_cascade = cv2.CascadeClassifier(os.path.join('haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join('haarcascade_eye.xml'))
    img = cv2.imread(user_img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(aligned_img_path[:-3]+'png', crop_img)
        return aligned_img_path
	return None

def get_embedding(img_filepath):
	return img_processor(img_filepath)

def get_best_matches_img_names(embedding):
	response = requests.post('http://localhost:8080/v2/best_matches', json={"embedding": embedding})
	logging.info('status_code: {}'.format(response.status_code))
	logging.info('response: {}'.format(response.json()))
	return []

def process_img(img_filepath):
	aligned_img = align_image(img_filepath)
	if aligned_img is None:
		logging.info("Can't find face")
		return []

	embedding = get_embedding(img_filepath)
    logging.info('embedding: {}'.format(embedding))
    best_matches_img_names = get_best_matches_img_names(embedding)
    logging.info('best_matches_img_paths: {}'.format(best_matches_img_names))
    return best_matches_img_names

# muzzle
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/get_image/<filename>')
def send_image(filename):
    return send_from_directory("users_images", filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'GET':
		return render_template('index.html')

	img_filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file = request.files['image']
    file.save(img_filepath)

    best_matches_img_names = process_img(img_filepath)
    return render_template('index.html', image_names=best_matches_img_names)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8081)
