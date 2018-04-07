import os
from PIL import Image
import tensorflow as tf
import numpy as np
import facenet
from annoy import AnnoyIndex
import loggging
import math

logging.basicConfig(level=logging.INFO)

DEBUG, DEBUG_SIZE = True, 500 # with debug flag we work only with DEBUG_SIZE images
ANNOY_SIZE = 128
ANNOY_TREES_COUNT = 10
IMG_SIZE = 160

def get_img_paths(raw_imgs_path):
    paths = []
    for dirpath, dnames, fnames in os.walk(raw_imgs_path):
        for fname in fnames:
            if fname.startswith('.'):
                continue
            paths.append(os.path.join(dirpath, fname))
    return paths

def resize_img(source_path, target_folder, img_size):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    _, img_name = os.path.split(source_path)

    img = Image.open(source_path)
    resized_img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    resized_img.save(os.path.join(target_folder, img_name))

def resize_imgs(source_folder, target_folder, img_size):
    img_paths = get_img_paths(source_folder)
    for img_path in img_paths:
        resize_img(img_path, target_folder, img_size)

def create_embeddings(img_aligned_scaled_path, trained_model_path, tmp_folder, embeddings_filename):
	embeddings_path = os.path.join(tmp_folder, embeddings_filename)
	aligned_imgs_paths = get_img_paths(img_aligned_scaled_path)
	if DEBUG:
		aligned_imgs_paths = aligned_imgs_paths[:DEBUG_SIZE]

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
    logging.info('create embeddings: {}'.format(embeddings_path))

def create_index(tmp_folder, embeddings_filename, index_filename):
	embeddings_path = os.path.join(tmp_folder, embeddings_filename)
	index_path = os.path.join(tmp_folder, index_filename)

	embeddings = None
	with open(embeddings_path, 'wb') as f:
        embeddings = pickle.load(f)
    if DEBUG:
    	embeddings = embeddings[:DEBUG_SIZE]

    annoy_index = AnnoyIndex(ANNOY_SIZE)
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)
    annoy_index.build(ANNOY_TREES_COUNT)
    annoy_index.save(index_path)
    logging.info('create index: {}'.format(index_path))

def is_aligned_scaled_imgs_exists(img_aligned_scaled_path):
	# temporary
	files_count = len(os.listdir(img_aligned_scaled_path))
	return files_count > 100

def is_embeddings_exists(tmp_folder, embeddings_filename):
	return os.path.isfile(os.path.join(tmp_folder, embeddings_filename))

def is_index_model_exists(tmp_folder, index_filename):
	return os.path.isfile(os.path.join(tmp_folder, index_filename))

if __name__ == '__main__':
	IMG_CELEBS_PATH = 'img_celebs'
	IMG_ALIGNED_PATH = 'img_aligned'
	IMG_ALIGNED_SCALED_PATH = 'img_aligned_scaled'
	TRAINED_MODEL_PATH = 'pretrained_model'
	TMP_FOLDER = 'find_index_server/tmp'
	INDEX_FILENAME = 'index.ann'
	EMBEDDINGS_FILENAME = 'embeddings.pickle'
	
	if DEBUG:
		INDEX_FILENAME += '.debug'
		EMBEDDINGS_FILENAME += '.debug'

	if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    if ((not is_embeddings_exists(TMP_FOLDER, EMBEDDINGS_FILENAME)) and
	    (not is_aligned_scaled_imgs_exists(IMG_ALIGNED_SCALED_PATH))):
		resize_imgs(IMG_ALIGNED_PATH, IMG_ALIGNED_SCALED_PATH)
	logging.info('scale aligned images - done')

	if not is_embeddings_exists(TMP_FOLDER, EMBEDDINGS_FILENAME):
		create_embeddings(IMG_ALIGNED_SCALED_PATH, TRAINED_MODEL_PATH, TMP_FOLDER, EMBEDDINGS_FILENAME)
	logging.info('create embeddings - done')

	if not is_index_model_exists(TMP_FOLDER, INDEX_FILENAME):
		create_index(TMP_FOLDER, EMBEDDINGS_FILENAME, INDEX_FILENAME)
	logging.info('create index - done')
