import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.keras.backend.set_floatx('float16')
tf.keras.backend.set_epsilon(1e-4)
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning

# Scikit-learn includes many helpful utilities


IMG_TARGET_SIZE = (299, 299)
IMG_TARGET_SHAPE = IMG_TARGET_SIZE + (3,)

# Download caption annotation files
annotation_folder = '/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
    annotation_zip = tf.keras.utils.get_file(
        'captions.zip',
        cache_subdir=os.path.abspath('.'),
        origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        extract=True
    )
    annotation_file = os.path.dirname(
        annotation_zip)+'/annotations/captions_train2014.json'
    os.remove(annotation_zip)

# Download image files
image_folder = '/train2014/'
if not os.path.exists(os.path.abspath('.') + image_folder):
    image_zip = tf.keras.utils.get_file(
        'train2014.zip',
        cache_subdir=os.path.abspath('.'),
        origin='http://images.cocodataset.org/zips/train2014.zip',
        extract=True
    )
    PATH = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
else:
    PATH = os.path.abspath('.') + image_folder


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.cast(img, tf.float16)
    return img, image_path


image_model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet')

if __name__=='__main__':
    with open('annotations/captions_train2014.json') as f:
        train_captions = json.loads(f.read())['images']

        train_filenames = [
            'train2014/' + caption['file_name']
            for caption in train_captions
        ]

        del train_captions

    train_filenames = sorted(set(train_filenames))

    def train_image_dataset_generator():
        for x in train_filenames:
            yield load_image(x)[0]


    train_image_dataset = tf.data.Dataset.from_generator(
        train_image_dataset_generator, output_types=tf.float16).batch(128)
    # train_image_dataset = train_image_dataset.map(
    #     load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(1)

    i = 0
    for img in train_image_dataset:
        features = image_model(img)
        path = train_filenames[i:i+features.shape[0]]
        i += features.shape[0]
        features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))

        for bf, p in zip(features, path):
            path_of_feature = p
            path_of_feature = 'image_embeddings'+path_of_feature[9:]+'.npy'
            with open(path_of_feature, 'wb') as f:
                np.save(f, bf.numpy())

    with open('annotations/captions_val2014.json') as f:
        val_captions = json.loads(f.read())['images']

        val_filenames = [
            'val2014/' + caption['file_name']
            for caption in val_captions
        ]

        del val_captions

    val_filenames = sorted(set(val_filenames))


    def val_image_dataset_generator():
        for x in val_filenames:
            yield load_image(x)[0]


    val_image_dataset = tf.data.Dataset.from_generator(
        val_image_dataset_generator, output_types=tf.float16).batch(128)
    # train_image_dataset = train_image_dataset.map(
    #     load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(1)

    i = 0
    for img in val_image_dataset:
        features = image_model(img)
        path = val_filenames[i:i+features.shape[0]]
        i += features.shape[0]
        features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))

        for bf, p in zip(features, path):
            path_of_feature = p
            path_of_feature = 'image_embeddings'+path_of_feature[7:]+'.npy'
            with open(path_of_feature, 'wb') as f:
                np.save(f, bf.numpy())