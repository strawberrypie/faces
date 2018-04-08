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
from tqdm import tqdm
import logging
import nmslib
from hnsw_index import HNSWIndex

logging.basicConfig(level=logging.INFO)

DEBUG_COUNT = 500
DEBUG_EXT = '.debug'
EMBEDDINGS_FILE = 'embeddings.pickle'
ANNOY_FILE = 'index.ann'
NMS_FILE = 'index.nms'
ANNOY_SIZE = 128
HNSW_SIZE = 128
ANNOY_TREES_COUNT = 10
BATCH_SIZE = 100
IMG_SIZE = 160

def get_filename_wo_ext(path):
        return os.path.basename(os.path.splitext(path)[0])

def resize_img(source_path, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    _, img_name = os.path.split(source_path)

    img = Image.open(source_path)
    resized_img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    resized_img.save(os.path.join(target_folder, img_name))

def resize_imgs(source_folder, target_folder, is_debug=False):
    img_paths = get_img_paths(source_folder)
    if is_debug:
        img_paths = img_paths[:DEBUG_COUNT]
    for img_path in tqdm(img_paths):
        resize_img(img_path, target_folder)

def align_image(user_img_path, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    _, img_name = os.path.split(user_img_path)
    aligned_img_path = os.path.join(target_folder, img_name)
    aligned_img_path = aligned_img_path[:-3]+ 'png'

    path_to_xml = os.path.join(cv2.__path__[0], 'data')
    face_cascade = cv2.CascadeClassifier(os.path.join(path_to_xml, 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(path_to_xml, 'haarcascade_eye.xml'))
    img = cv2.imread(user_img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))

        cv2.imwrite(aligned_img_path, crop_img)
        return aligned_img_path
    return None

def aligned_imgs(source_folder, target_folder, is_debug=False):
    img_paths = get_img_paths(source_folder)
    if is_debug:
        img_paths = img_paths[:DEBUG_COUNT]
    for img_path in tqdm(img_paths):
        align_image(img_path, target_folder)

def get_img_paths(raw_imgs_path, avail_filenames=None):
    paths = []
    for dirpath, dnames, fnames in os.walk(raw_imgs_path):
        for fname in fnames:
            if fname.startswith('.'):
                continue
            if (avail_filenames is None) or (os.path.splitext(fname)[0] in avail_filenames):
                paths.append(os.path.join(dirpath, fname))
    paths = sorted(paths)
    if avail_filenames is not None:
        for i, path in enumerate(paths):
            logging.debug('raw img mapping: {}: {}'.format(i, path))
    return paths

def get_index(aligned_user_img_path, trained_model_path):
    embeddings_arr = None
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(trained_model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            images = facenet.load_data([aligned_user_img_path], False, False, IMG_SIZE)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            embeddings_arr = sess.run(embeddings, feed_dict=feed_dict)
    return embeddings_arr[0]

def index_exists(tmp_dir, index_label, is_debug):
    assert(os.path.isdir(tmp_dir))
    if index_label == 'annoy':
        index_filename = ANNOY_FILE
    elif index_label == 'nms':
        index_filename = NMS_FILE
    elif index_label == 'hnsw':
        return False
    else:
        raise NotImlemented('not implemented index')
    if is_debug:
        index_filename += DEBUG_EXT
    return os.path.isfile(os.path.join(tmp_dir, index_filename))

def create_annoy_index(tmp_dir, embeddings, is_debug):
    annoy_index = AnnoyIndex(ANNOY_SIZE)
    annoy_filename = ANNOY_FILE
    if is_debug:
        embeddings = embeddings[:DEBUG_COUNT]
        annoy_filename += DEBUG_EXT
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)
    annoy_index.build(ANNOY_TREES_COUNT)
    annoy_index.save(os.path.join(tmp_dir, annoy_filename))
    logging.info('saved annoy index: {}'.format(os.path.join(tmp_dir, annoy_filename)))

def create_nms_index(tmp_dir, embeddings, is_debug):
    nms_filename = NMS_FILE
    if is_debug:
        embeddings = embeddings[:DEBUG_COUNT]
        nms_filename += DEBUG_EXT
    nms_index = nmslib.init(method='hnsw', space='cosinesimil')
    nms_index.addDataPointBatch(np.array(embeddings))
    nms_index.createIndex({'post': 2}, print_progress=True)
    nms_index.saveIndex(os.path.join(tmp_dir, nms_filename))
    logging.info('saved nms index: {}'.format(os.path.join(tmp_dir, nms_filename)))

hnsw_index = None

def create_hnsw_index(tmp_dir, embeddings, is_debug):
    global hnsw_index
    hnsw_index = HNSWIndex(HNSW_SIZE)
    if is_debug:
        embeddings = embeddings[:DEBUG_COUNT]
    hnsw_index.add_items(np.array(embeddings), np.arange(len(embeddings)))

def create_index(tmp_dir, embeddings, index_label, is_debug):
    if index_label == 'annoy':
        create_annoy_index(tmp_dir, embeddings, is_debug)
    elif index_label == 'nms':
        create_nms_index(tmp_dir, embeddings, is_debug)
    elif index_label == 'hnsw':
        create_hnsw_index(tmp_dir, embeddings, is_debug)
    else:
        raise NotImlemented('not implemented index')

def get_best_matches_annoy_idxs(tmp_dir, user_img_index, is_debug, count=3):
    annoy_index = AnnoyIndex(ANNOY_SIZE)
    annoy_filepath = os.path.join(tmp_dir, ANNOY_FILE)
    if is_debug:
        annoy_filepath += DEBUG_EXT
    annoy_index.load(annoy_filepath)
    best_matches_idxs = annoy_index.get_nns_by_vector(user_img_index, count)
    return best_matches_idxs

def get_best_matches_nms_idxs(tmp_dir, user_img_index, is_debug, count=3):
    nms_index_filepath = os.path.join(tmp_dir, NMS_FILE)
    if is_debug:
        nms_index_filepath += DEBUG_EXT
    nms_index = nmslib.init(method='hnsw', space='cosinesimil')
    nms_index.loadIndex(nms_index_filepath)
    best_matches_idxs, distances = nms_index.knnQuery(np.array(user_img_index), k=count)
    return best_matches_idxs

def get_best_matches_hnsw_idxs(tmp_dir, user_img_index, is_debug, count=3):
    best_matches_idxs, distances = hnsw_index.knn_query(np.array([user_img_index]), k=count)
    return best_matches_idxs[0]

def get_best_matches_idxs(tmp_dir, user_img_index, index_label, is_debug, count=3):
    if index_label == 'annoy':
        return get_best_matches_annoy_idxs(tmp_dir, user_img_index, is_debug, count)
    elif index_label == 'nms':
        return get_best_matches_nms_idxs(tmp_dir, user_img_index, is_debug, count)
    elif index_label == 'hnsw':
        return get_best_matches_hnsw_idxs(tmp_dir, user_img_index, is_debug, count)
    raise NotImlemented('not implemented index') 

def get_aligned_img_idxs(aligned_imgs_path, is_debug):
    aligned_imgs_paths = get_img_paths(aligned_imgs_path)
    if is_debug:
        aligned_imgs_paths = aligned_imgs_paths[:DEBUG_COUNT]
    img_idxs = [get_filename_wo_ext(x) for x in aligned_imgs_paths]
    return img_idxs

def get_embeddings(aligned_imgs_path, trained_model_path, tmp_dir, is_debug):
    assert(os.path.isdir(tmp_dir))
    embeddings_filename = EMBEDDINGS_FILE
    if is_debug:
        embeddings_filename += DEBUG_EXT
    embeddings_path = os.path.join(tmp_dir, embeddings_filename)
    if os.path.isfile(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            return pickle.load(f)

    aligned_imgs_paths = get_img_paths(aligned_imgs_path)
    if is_debug:
        aligned_imgs_paths = aligned_imgs_paths[:DEBUG_COUNT]
    img_idxs = [get_filename_wo_ext(x) for x in aligned_imgs_paths]

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
                pickle.dump((emb_array, img_idxs), f)
            logging.info('saved embeddings: {}'.format(embeddings_path))

            for i in range(len(emb_array)):
                logging.debug('embedding: {}: {}'.format(i, aligned_imgs_paths[i]))

            return emb_array, img_idxs

def show_imgs(user_img_path, best_matches_img_paths):
    fig = plt.figure()
    a = fig.add_subplot(1, len(best_matches_img_paths) + 1, 1)
    # show user image
    img = mpimg.imread(user_img_path)
    imgplot = plt.imshow(img)
    # show best matches
    for i in range(2, len(best_matches_img_paths) + 2):
        a = fig.add_subplot(1, len(best_matches_img_paths) + 1, i)
        img = mpimg.imread(best_matches_img_paths[i-2])
        imgplot = plt.imshow(img)
    plt.show()

def show_sim_celebrities(aligned_imgs_path, raw_imgs_path, trained_model_path,
                         user_img_path, tmp_dir, is_debug, index_label):
    # init
    if not index_exists(tmp_dir, index_label, is_debug):
        logging.info('creating index')
        embeddings, img_idxs = get_embeddings(aligned_imgs_path, trained_model_path, tmp_dir, is_debug)
        create_index(tmp_dir, embeddings, index_label, is_debug)
    else:
        logging.info('index exists')
        img_idxs = get_aligned_img_idxs(aligned_imgs_path, is_debug)

    available_indexes = set(img_idxs)
    raw_img_paths = get_img_paths(raw_imgs_path, available_indexes)

    # main pipeline:
    align_user_img_path = align_image(user_img_path, tmp_dir)
    user_img_index = get_index(align_user_img_path, trained_model_path)
    best_matches_idxs = get_best_matches_idxs(tmp_dir, user_img_index, index_label, is_debug)
    logging.debug('best_matches idxs: {}'.format(best_matches_idxs))
    best_matches_img_paths = np.array(raw_img_paths)[best_matches_idxs]
    logging.info('best_matches_img_paths: {}'.format(best_matches_img_paths))
    show_imgs(user_img_path, best_matches_img_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resize_imgs', action='store_true')
    parser.add_argument('--source_folder', type=str, help='path to aligned celebrities imgs')
    parser.add_argument('--target_folder', type=str, help='path to aligned resized celebrities imgs')
    # need for generating indexes. Not needed when model wouldn't change
    parser.add_argument('--aligned_imgs_path', type=str, help='path to aligned celebrities imgs')
    parser.add_argument('--raw_imgs_path', type=str, help='path to celebrities imgs')
    # get from https://github.com/davidsandberg/facenet
    parser.add_argument('--trained_model_path', type=str, help='path to pretrained model')
    parser.add_argument('--user_img_path', type=str, help='path to raw img to estimate')
    parser.add_argument('--worker_dir', type=str, help='dir for tmp files', default='~/tmp')
    parser.add_argument('--index_label', type=str, help='index label, available: annoy/nms/hnsw', default='annoy')
    args = parser.parse_args()

    if args.resize_imgs:
        aligned_imgs(args.source_folder, args.target_folder, args.debug)
    else:
        show_sim_celebrities(args.aligned_imgs_path, args.raw_imgs_path, args.trained_model_path,
                args.user_img_path, args.worker_dir, args.debug, args.index_label)

# python find_sim_celebrities_debug.py --resize_imgs --source_folder=../aligned/aligned --target_folder=../aligned_sized --debug
# python find_sim_celebrities_debug.py --aligned_imgs_path=../aligned_sized --raw_imgs_path=../raw/raw --trained_model_path=../pretrained_model --user_img_path=../user_imgs/Angelina-Jolie.jpg --worker_dir=../tmp --debug --index_label
