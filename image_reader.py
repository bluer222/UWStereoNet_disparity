import os
import glob
import collections
import tensorflow as tf


def preprocess(img):
    """Normalize float32 image in [0, 1] to [-1, 1]."""
    return img * 2 - 1


def _decode_and_preprocess(path):
    contents = tf.io.read_file(path)
    image = tf.image.decode_png(contents, channels=3, dtype=tf.uint8)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return preprocess(image)


def _load_pair(left_path, right_path, height, width, dataset):
    left = _decode_and_preprocess(left_path)
    right = _decode_and_preprocess(right_path)
    total = tf.concat([left, right], axis=-1)
    if dataset == 'cityscapes':
        total = tf.image.resize(total, [512, 1024])
    cropped = tf.image.random_crop(total, [height, width, 6])
    left, right = tf.split(cropped, [3, 3], axis=-1)
    left.set_shape([height, width, 3])
    right.set_shape([height, width, 3])
    return left, right


def list_stereo_pairs(left_dir, right_dir):
    left_paths = sorted(glob.glob(os.path.join(left_dir, '*.png')))
    right_paths = sorted(glob.glob(os.path.join(right_dir, '*.png')))
    if len(left_paths) == 0:
        raise FileNotFoundError(f'No PNGs found in {left_dir}')
    if len(left_paths) != len(right_paths):
        raise ValueError(
            f'Mismatched stereo counts: {len(left_paths)} left vs {len(right_paths)} right')
    return left_paths, right_paths


def create_dataset(left_dir, right_dir, height, width, dataset, batch_size,
                   shuffle=True, repeat=True):
    left_paths, right_paths = list_stereo_pairs(left_dir, right_dir)
    ds = tf.data.Dataset.from_tensor_slices((left_paths, right_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(left_paths), 1024), reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.map(
        lambda l, r: _load_pair(l, r, height, width, dataset),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(left_paths)


def load_examples(dirs, size, dataset, batch_size):
    """
    Compatibility wrapper around tf.data.

    Returns a namedtuple with a dataset iterator tensors for a single next batch,
    plus count/batch_size metadata. Prefer create_dataset() for new code.
    """
    Examples = collections.namedtuple('Examples', 'lefts, rights, count, batch_size, dataset')
    ds, count = create_dataset(
        dirs[0], dirs[1], size[0], size[1], dataset, batch_size,
        shuffle=True, repeat=True)
    it = iter(ds)
    left, right = next(it)
    return Examples(lefts=left, rights=right, count=count, batch_size=batch_size, dataset=ds)
