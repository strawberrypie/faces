from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# export PYTHONPATH=[path_to_facenet]/facenet/src
import facenet

import tensorflow as tf
import numpy as np
import argparse
import lfw
import os
import sys
import math
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from annoy import AnnoyIndex
from PIL import Image
import cv2

EMBEDDINGS_FILE = 'embeddings.pickle'
ANNOY_FILE = 'index.ann'
ANNOY_SIZE = 128
ANNOY_TREES_COUNT = 10
BATCH_SIZE = 100
IMG_SIZE = 160

def resize_img(source_path, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    _, img_name = os.path.split(source_path)

    img = Image.open(source_path)
    resized_img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    resized_img.save(os.path.join(target_folder, img_name))

def resize_imgs(source_folder, target_folder):
    img_paths = get_img_paths(source_folder)
    for img_path in img_paths:
        resize_img(img_path, target_folder)

def align_image(user_img_path, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    _, img_name = os.path.split(user_img_path)
    aligned_img_path = os.path.join(target_folder, img_name)

    path_to_xml = '/Users/romanmarakulin/anaconda3/lib/python3.6/site-packages/cv2/data'
    face_cascade = cv2.CascadeClassifier(os.path.join(path_to_xml, 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(path_to_xml, 'haarcascade_eye.xml'))
    img = cv2.imread(user_img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces:
        x, y, w, h = faces[0]
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))

        cv2.imwrite(aligned_img_path[:-3]+'png', crop_img)
        return aligned_img_path
    return None

def get_img_paths(raw_imgs_path):
    paths = []
    for dirpath, dnames, fnames in os.walk(raw_imgs_path):
        for fname in fnames:
            if fname.startswith('.'):
                continue
            paths.append(os.path.join(dirpath, fname))
    return paths

def get_index(align_user_img_path, trained_model_path):
    embeddings_arr = None
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(trained_model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            images = facenet.load_data([img_path], False, False, IMG_SIZE)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            embeddings_arr = sess.run(embeddings, feed_dict=feed_dict)
    return embeddings_arr[0]

def annoy_index_exists(tmp_dir):
    assert(os.path.isdir(tmp_dir))
    return os.path.isfile(os.path.join(tmp_dir, ANNOY_FILE))

def create_annoy_index(tmp_dir, embeddings):
    annoy_index = AnnoyIndex(ANNOY_SIZE)
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)
    annoy_index.build(ANNOY_TREES_COUNT)
    annoy_index.save(os.path.join(tmp_dir, ANNOY_FILE))

def get_best_matches_idxs(tmp_dir, user_img_index, count=1):
    annoy_index = AnnoyIndex(ANNOY_SIZE)
    annoy_index.load(os.path.join(tmp_dir, ANNOY_FILE))
    best_matches_idxs = annoy_index.get_nns_by_vector(user_img_index, count)
    return best_matches_idxs

def get_embeddings(aligned_imgs_path, trained_model_path, tmp_dir):
    assert(os.path.isdir(tmp_dir))
    embeddings_path = os.path.join(tmp_dir, EMBEDDINGS_FILE)
    if os.path.isfile(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            return pickle.load(f)

    aligned_imgs_paths = get_img_paths(aligned_imgs_path)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(trained_model_path)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            embedding_size = embeddings.get_shape()[1]
            nrof_images = len(aligned_imgs_paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / BATCH_SIZE))
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches):
                start_index = i * BATCH_SIZE
                end_index = min((i+1)*BATCH_SIZE, nrof_images)
                paths_batch = aligned_imgs_paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, IMG_SIZE)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            with open(embeddings_path, 'wb') as f:
                pickle.dump(emb_array, f)
            return emb_array

def show_imgs(user_img_path, best_matches_img_paths):
    fig = plt.figure()
    a = fig.add_subplot(len(best_matches_img_paths) + 1, 1, 1)
    # show user image
    img = mpimg.imread(user_img_path)
    imgplot = plt.imshow(img)
    # show best matches
    for i in range(2, len(best_matches_img_paths) + 2):
        a = fig.add_subplot(len(best_matches_img_paths) + 1, 1, i)
        img = mpimg.imread(best_matches_img_paths[i-2])
        imgplot = plt.imshow(img)
    plt.show()

def show_sim_celebrities(aligned_imgs_path, raw_imgs_path, trained_model_path, user_img_path, tmp_dir):
    # init
    if not annoy_index_exists(tmp_dir):
        embeddings = get_embeddings(aligned_imgs_path, trained_model_path, tmp_dir)
        create_annoy_index(tmp_dir, embeddings)
    raw_img_paths = get_img_paths(raw_imgs_path)

    # main pipeline:
    align_user_img_path = align_image(user_img_path, tmp_dir)
    user_img_index = get_index(align_user_img_path, trained_model_path)
    best_matches_idxs = get_best_matches_idxs(tmp_dir, user_img_index)
    best_matches_img_paths = raw_img_paths[best_matches_idxs]
    show_imgs(user_img_path, best_matches_img_paths)

if __name__ == '__main__':
    test_samples()
    sys.exit(0)

    parser = argparse.ArgumentParser()
    # need for generating indexes. Not needed when model wouldn't change
    parser.add_argument('--aligned_imgs_path', type=str, help='path to aligned celebrities imgs')
    parser.add_argument('--raw_imgs_path', type=str, help='path to celebrities imgs')
    parser.add_argument('--trained_model_path', type=str, help='path to pretrained model')
    parser.add_argument('--user_img_path', type=str, help='path to raw img to estimate')
    parser.add_argument('--worker_dir', type=str, help='dir for tmp files', default='~/tmp')
    args = parser.parse_args()

    show_sim_celebrities(args.aligned_imgs_path, args.raw_imgs_path, args.trained_model_path,
            args.user_img_path, args.worker_dir)
