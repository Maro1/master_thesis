from segmentation_mask import AlphaPredictor
from scipy.spatial.transform import Rotation
from objectron.schema import features
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import cv2
import sys
import argparse
import shutil

sys.path.append('..')


os.environ['CURL_CA_BUNDLE'] = "/etc/ssl/certs/ca-bundle.crt"

objectron_buckett = 'gs://objectron'
NUM_PARALLEL_CALLS = 4
WIDTH = 480
HEIGHT = 640
NUM_CHANNELS = 3
# The 3D bounding box has 9 vertices, 0: is the center, and the 8 vertices of the 3D box.
NUM_KEYPOINTS = 9
BATCH_SIZE = 4

NUM_FRAMES = 100


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Objectron Pre-Process',
        description='Pre-processes Objectron dataset for use with GET3D')
    parser.add_argument('output')
    parser.add_argument('--category', type=str, default='book')

    return parser.parse_args()


def parse_tfrecord(example):
    context, data = tf.io.parse_single_sequence_example(
        example,
        sequence_features=features.SEQUENCE_FEATURE_MAP,
        context_features=features.SEQUENCE_CONTEXT_MAP
    )

    # Number of frames in this video sequence
    num_examples = context['count']
    # The unique sequence id (class/batch-i/j) useful for visualization/debugging
    video_id = context['sequence_id']

    rand = tf.cast(tf.linspace(0, tf.cast(num_examples, tf.int32) -
                   tf.constant(1, dtype=tf.int32), NUM_FRAMES), tf.int64)

    data['frame_ids'] = rand
    # Grab 4 random frames from the sequence and decode them
    for i in range(NUM_FRAMES):
        id = rand[i]
        image_tag = 'image-{}'.format(i)
        data[image_tag] = data[features.FEATURE_NAMES['IMAGE_ENCODED']][id]
        data[image_tag] = tf.image.decode_png(data[image_tag], channels=3)
        data[image_tag].set_shape([640, 480, 3])

    return context, data


def parse(example):
    """Parses a single tf.Example and decode the `png` string to an array."""
    data = tf.io.parse_single_example(example, features=features.FEATURE_MAP)
    data['image'] = tf.image.decode_png(
        data[features.FEATURE_NAMES['IMAGE_ENCODED']], channels=NUM_CHANNELS)
    data['image'].set_shape([HEIGHT, WIDTH, NUM_CHANNELS])
    return data


def augment(data):
    return data


def normalize(data):
    """Convert `image` from [0, 255] -> [-1., 1.] floats."""
    data['image'] = tf.cast(data['image'], tf.float32) * (2. / 255.) - 1.0
    return data


def load_tf_record(input_record):
    dataset = tf.data.TFRecordDataset(input_record)

    dataset = dataset.map(parse, num_parallel_calls=NUM_PARALLEL_CALLS)\
                     .map(augment, num_parallel_calls=NUM_PARALLEL_CALLS)\
                     .map(normalize, num_parallel_calls=NUM_PARALLEL_CALLS)
    # Our TF.records are shuffled in advance. If you re-generate the dataset from the video files, you'll need to
    # shuffle your examples. Keep in mind that you cannot shuffle the entire datasets using dataset.shuffle, since
    # it will be very slow.
    dataset = dataset.shuffle(100)\
                     .repeat()\
                     .batch(BATCH_SIZE)\
                     .prefetch(buffer_size=10)
    return dataset


def find_angles(R):
    """Converts from Euler angles to rotation and elevation"""
    euler = Rotation.from_matrix(R).as_euler('YXZ', degrees=True)

    elevation = -euler[1]
    rotation = -euler[0]
    if rotation < 0:
        rotation = 360 + rotation

    return rotation, elevation


def pre_process(args, split, count):
    """Pre-processes the Objectron records of a specific category for use with GET3D"""
    alpha_predictor = AlphaPredictor()
    bucket_str = '/v1/sequences/' + args.category + \
        '/' + args.category + '_' + split + '*'
    shards = tf.io.gfile.glob(objectron_buckett + bucket_str)

    dataset = tf.data.TFRecordDataset(shards)
    dataset = dataset.map(parse_tfrecord)

    manifest_path = os.path.join(os.path.abspath(args.output), split + '.txt')

    it = dataset.__iter__()
    while True:
        try:
            optional = it.get_next_as_optional()
        except:
            continue

        if not optional.has_value():
            break
        context, data = optional.get_value()

        count += 1
        with open(manifest_path, 'a') as f:
            f.write('{0:04}'.format(count) + '\n')

        # Disregard objects with fewer frames than required
        num_frames = context['count']
        if num_frames < NUM_FRAMES:
            continue

        rotation_angle_list = list()
        elevation_angle_list = list()

        for i in range(NUM_FRAMES):
            id = data['frame_ids'][i]
            img_folder = os.path.join(os.path.abspath(
                args.output), 'img', args.category, '{0:04}'.format(count))

            # Disregard records with more than 1 object
            if data['instance_num'][id].numpy()[0] > 1:
                if os.path.exists(img_folder):
                    shutil.rmtree(img_folder)
                break

            # Disregard records with non-visible objects
            if data['object/visibility'].values.numpy()[id] != 1:
                if os.path.exists(img_folder):
                    shutil.rmtree(img_folder)
                break

            # Find segmentation mask
            image = data['image-{}'.format(i)].numpy()
            alpha = alpha_predictor.find_mask(image, args.category)
            if alpha is None:
                print('Unable to find mask for object ' +
                      str(count) + ' at frame ' + str(i) + '.')
                if os.path.exists(img_folder):
                    shutil.rmtree(img_folder)
                break

            alpha = alpha.astype(np.uint8) * 255

            image = cv2.bitwise_and(image, image, mask=alpha)

            rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = alpha

            # Crop and resize image
            if rgba.shape[0] > rgba.shape[1]:
                rgba = rgba[80:HEIGHT-80, :]
            else:
                rgba = rgba[:, 80:HEIGHT-80]

            rgba = cv2.resize(rgba, (512, 512))

            img_filename = '{0:03}'.format(i + 1) + '.png'
            os.makedirs(img_folder, exist_ok=True)
            cv2.imwrite(os.path.join(img_folder, img_filename), rgba)

            object_orientation = data['object/orientation'].values.numpy()[
                id*9:id*9+9].reshape(3, 3)
            object_euler = Rotation.from_matrix(
                object_orientation).as_euler('xyz', degrees=True)

            rotation = object_euler[0]
            elevation = object_euler[1]
            if rotation < 0:
                rotation = 360 + rotation
            if elevation < 0:
                elevation = 360 + elevation

            rotation_angle_list.append(rotation)
            elevation_angle_list.append(elevation)

        if len(rotation_angle_list) != NUM_FRAMES:
            continue

        camera_folder = os.path.join(os.path.abspath(
            args.output), 'camera', args.category, '{0:04}'.format(count))
        os.makedirs(camera_folder, exist_ok=True)

        np.save(os.path.join(camera_folder, 'rotation'), rotation_angle_list)
        np.save(os.path.join(camera_folder, 'elevation'), elevation_angle_list)


if __name__ == '__main__':
    args = parse_args()
    count = 0
    pre_process(args, 'train', count)
    pre_process(args, 'test', count)
