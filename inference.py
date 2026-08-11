import os
import argparse
import numpy as np
import png
import tensorflow as tf
from PIL import Image

from image_reader import preprocess


parser = argparse.ArgumentParser()
parser.add_argument("--export_dir", default='export/dropuwstereo_disp_cityscapes/', help="path to folder containing export files")
parser.add_argument("--output_dir", default='prediction/dropuwstereo_disp_cityscapes/', help="path to folder to save results")
parser.add_argument("--left_dir", default='data/test/left/', help="path to folder containing left-view images")
parser.add_argument("--right_dir", default='data/test/right/', help="path to folder containing right-view images")
parser.add_argument("--gpu", type=str, default='0')

a = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu


def load_image(path):
    content = tf.io.read_file(path)
    image = tf.image.decode_png(content, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = preprocess(image)
    return tf.expand_dims(image, axis=0)


def main():
    left_out_dir = os.path.join(a.output_dir, 'left/')
    right_out_dir = os.path.join(a.output_dir, 'right/')
    os.makedirs(left_out_dir, exist_ok=True)
    os.makedirs(right_out_dir, exist_ok=True)

    if a.export_dir is None:
        raise Exception("checkpoint required for test mode")

    print("loading exported model from: {}".format(a.export_dir))
    model = tf.saved_model.load(a.export_dir)
    predict_fn = model.signatures['serving_default']

    print('running...')
    filenames = sorted(os.listdir(a.left_dir))
    for item in filenames:
        if not item.lower().endswith('.png'):
            continue
        l_path = os.path.join(a.left_dir, item)
        r_path = os.path.join(a.right_dir, item)
        # Ensure shape divisible by 32 as noted in README.
        shape = Image.open(l_path).size
        if shape[0] % 32 != 0 or shape[1] % 32 != 0:
            print('skipping {} (shape {} not divisible by 32)'.format(item, shape))
            continue

        left = load_image(l_path)
        right = load_image(r_path)
        result = predict_fn(left=left, right=right)
        left_disp = np.squeeze(result['left_disp_pred'].numpy())
        right_disp = np.squeeze(result['right_disp_pred'].numpy())

        with open(os.path.join(left_out_dir, item), 'wb') as f:
            writer = png.Writer(width=left_disp.shape[1], height=left_disp.shape[0], greyscale=True, bitdepth=16)
            writer.write(f, left_disp.tolist())

        with open(os.path.join(right_out_dir, item), 'wb') as f:
            writer = png.Writer(width=right_disp.shape[1], height=right_disp.shape[0], greyscale=True, bitdepth=16)
            writer.write(f, right_disp.tolist())

        print('writing: {}'.format(item))


if __name__ == "__main__":
    main()
