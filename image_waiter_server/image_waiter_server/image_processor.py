import os
import time
import cv2
import logging
from facenet.src import facenet
import tensorflow as tf
import numpy as np
from image_waiter_server.index_requester import IndexRequester, DummyIndexRequester

class ImageProcessor(object):
    def __init__(self, aligned_img_folder='../aligned_sized/', aligned_usr_img_folder='../tmp/',
        aligned_img_size=160, pretrained_model='../pretrained_model', debug=False):
        self.debug = debug
        self._aligned_usr_img_folder = aligned_usr_img_folder
        self._aligned_img_folder = aligned_img_folder
        self._aligned_img_size = aligned_img_size
        self._aligned_img_ext = 'png'
        self._raw_img_ext = 'jpg'
        self._img_filenames = self.get_img_filenames()
        
        path_to_cv2_xmls = os.path.join(cv2.__path__[0], 'data')
        self._face_cascade = cv2.CascadeClassifier(os.path.join(path_to_cv2_xmls, 'haarcascade_frontalface_default.xml'))
        self._eye_cascade = cv2.CascadeClassifier(os.path.join(path_to_cv2_xmls, 'haarcascade_eye.xml'))

        # load model
        self._sess = tf.Session()
        with self._sess.as_default():
            facenet.load_model(pretrained_model)
        self._images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self._embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self._phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # index service
        index_requester_url = 'http://localhost' if self.debug else 'http://faces_index_search_1'
        self._index_requester = IndexRequester(url=index_requester_url, port=8081)

    def get_img_filenames(self):
        paths = []
        for dirpath, dnames, fnames in os.walk(self._aligned_img_folder):
            for fname in fnames:
                if fname.startswith('.'):
                    continue
                fname = fname[:-3] + self._raw_img_ext
                paths.append(fname)
        return np.sort(paths)

    def _align_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            crop_img = img[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (self._aligned_img_size, self._aligned_img_size))
            return crop_img
        return None

    def align_image(self, img_filepath):
        start_time = time.time()

        _, img_name = os.path.split(img_filepath)
        aligned_img_path = os.path.join(self._aligned_usr_img_folder, img_name)
        aligned_img_path = aligned_img_path[:-3]+'.'+self._aligned_img_ext

        img = cv2.imread(img_filepath)

        aligned_img = self._align_image(img)
        if aligned_img is None:
            return None

        cv2.imwrite(aligned_img_path, aligned_img)

        logging.warning('align_image time: {}s'.format(time.time() - start_time))
        logging.info('aligned_img_path: {}'.format(aligned_img_path))
        return aligned_img_path

    def get_embedding(self, img_path):
        start_time = time.time()
        images = facenet.load_data([img_path], False, False, self._aligned_img_size)
        feed_dict = {self._images_placeholder: images, self._phase_train_placeholder: False}
        embeddings_arr = self._sess.run(self._embeddings, feed_dict=feed_dict)
        logging.warning('get_embedding time: {}s'.format(time.time() - start_time))
        return embeddings_arr[0]

    def get_best_matches_img_names(self, embedding, count):
        start_time = time.time()
        best_idxs, distances = self._index_requester.get_best_idxs(embedding, count)
        logging.warning('get best matches time: {}s'.format(time.time() - start_time))
        if best_idxs is None:
            logging.error("Can't get best idxs from index service")
            return []
        logging.info('best_idxs: {}\ndistances: {}'.format(best_idxs, distances))

        try:
            result = self._img_filenames[best_idxs]
        except IndexError:
            logging.error("Can't find images with indexes: {}".format(best_idxs))
            result = []
        return result

    def get_best_matches_imgs(self, img_filepath, count=1):
        if not os.path.isfile(img_filepath):
            logging.error("Can't find img: {}".format(img_filepath))
            return []
        if (type(count) is not int) or (count < 1):
            logging.error("count should be int and > 0".format(count))
            return []

        try:
            aligned_img_filepath = self.align_image(img_filepath)
        except cv2.error as e:
            aligned_img_filepath = None
        if aligned_img_filepath is None:
            logging.error("Can't find face on img: {}".format(img_filepath))
            return []

        embedding = self.get_embedding(aligned_img_filepath)

        best_matches_img_names = self.get_best_matches_img_names(embedding, count)
        logging.info('best_matches_img_paths: {}'.format(best_matches_img_names))
        return best_matches_img_names
