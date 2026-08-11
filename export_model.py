import os
import argparse
import tensorflow as tf

from model import StereoNet


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", default='checkpoint/dropuwstereo_disp_cityscapes/', help="path to folder containing checkpoint")
parser.add_argument("--export_dir", default='export/dropuwstereo_disp_cityscapes/', help="path to folder to save export files")
parser.add_argument("--max_num_disparity", type=int, default=192, help="maximum value for disparity")
parser.add_argument("--height", type=int, default=256, help="dummy build height (divisible by 32)")
parser.add_argument("--width", type=int, default=512, help="dummy build width (divisible by 32)")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--monochrome', dest='monochrome', action='store_true',
                    help="export a model that expects single-channel grayscale inputs")

a = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu


def make_inference_module(model, max_num_disparity, monochrome):
    channels = 1 if monochrome else 3

    class InferenceModule(tf.Module):
        """SavedModel wrapper: left/right float images in [-1, 1] -> uint16 disp maps."""

        def __init__(self):
            super().__init__()
            self.model = model
            self.max_num_disparity = max_num_disparity

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, None, None, channels], dtype=tf.float32, name='left'),
            tf.TensorSpec(shape=[1, None, None, channels], dtype=tf.float32, name='right'),
        ])
        def predict(self, left, right):
            outputs = self.model([left, right], training=False)
            left_disp_pred = tf.cast(outputs['left_refined_disp'] * 256.0, tf.uint16)
            right_disp_pred = tf.cast(outputs['right_refined_disp'] * 256.0, tf.uint16)
            return {
                'left_disp_pred': left_disp_pred,
                'right_disp_pred': right_disp_pred,
            }

    return InferenceModule()


def main():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    in_channels = 1 if a.monochrome else 3
    model = StereoNet(max_num_disparity=a.max_num_disparity)
    dummy = tf.zeros([1, a.height, a.width, in_channels], dtype=tf.float32)
    _ = model([dummy, dummy], training=False)

    ckpt = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(a.checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f'No checkpoint found in {a.checkpoint_dir}')
    print('loading model from: {}'.format(latest))
    ckpt.restore(latest).expect_partial()

    os.makedirs(a.export_dir, exist_ok=True)
    module = make_inference_module(model, a.max_num_disparity, a.monochrome)
    _ = module.predict(dummy, dummy)

    print('export model to: {} (monochrome={})'.format(a.export_dir, a.monochrome))
    tf.saved_model.save(
        module,
        a.export_dir,
        signatures={'serving_default': module.predict})


if __name__ == "__main__":
    main()
