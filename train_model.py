import argparse
import os
import logging
import tensorflow as tf

from image_reader import create_dataset
from model import StereoNet
from utils import deprocess


parser = argparse.ArgumentParser()
parser.add_argument("--left_dir", default='data/cityscapes/train/left/', help="path to folder containing left-view training images")
parser.add_argument("--right_dir", default='data/cityscapes/train/right/', help="path to folder containing right-view training images")
parser.add_argument("--left_val_dir", default='data/cityscapes/val/left/', help="path to folder containing left-view validation images")
parser.add_argument("--right_val_dir", default='data/cityscapes/val/right/', help="path to folder containing right-view validation images")
parser.add_argument("--checkpoint_dir", default='checkpoint/dropuwstereo_disp_cityscapes/', help="where to put checkpoints files")
parser.add_argument("--summary_dir", default='summary/dropuwstereo_disp_cityscapes/', help="where to put summary files")
parser.add_argument("--resume_dir", default=None, help="directory with checkpoint to resume training from")

parser.add_argument("--num_steps", type=int, default=100000, help="number of training steps")
parser.add_argument("--summary_freq", type=int, default=15, help="frequency to update summaries")
parser.add_argument("--schedule_freq", type=int, default=50000, help="frequency to half learning rate")
parser.add_argument("--print_summary_freq", type=int, default=50, help="frequency to print summary")
parser.add_argument("--save_freq", type=int, default=10000, help="frequency to save model")

parser.add_argument("--w1", type=float, default=0.3, help="weight for initial disparity loss")
parser.add_argument("--w2", type=float, default=0.7, help="weight for refined disparity loss")
parser.add_argument("--beta1", type=float, default=0.8, help="initial disparity loss")
parser.add_argument("--beta2", type=float, default=0.01, help="initial disparity loss")
parser.add_argument("--beta3", type=float, default=0.001, help="initial disparity loss")
parser.add_argument("--gamma1", type=float, default=0.8, help="refined disparity loss")
parser.add_argument("--gamma2", type=float, default=0.02, help="refined disparity loss")
parser.add_argument("--gamma3", type=float, default=0.002, help="refined disparity loss")
parser.add_argument("--height", type=int, default=256, help="crop images to this height")
parser.add_argument("--width", type=int, default=512, help="crop images to this width")
parser.add_argument("--max_num_disparity", type=int, default=192, help="maximum value for disparity")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate for adam")
parser.add_argument("--dataset", type=str, default='cityscapes', choices=["kitti", "cityscapes"])
parser.add_argument("--gpu", type=str, default='0', help="which gpu to use")
parser.add_argument('--is_val', dest='is_val', action='store_true', help="show validation loss")

a = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu

os.makedirs(a.summary_dir, exist_ok=True)
os.makedirs(a.checkpoint_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(a.summary_dir, 'parameters.log'), level=logging.DEBUG)
adict = vars(a)
for key in sorted(adict.keys()):
    logging.info('{0}:{1}'.format(key, adict[key]))

WEIGHTS_LIST = [a.beta1, a.beta2, a.beta3, a.gamma1, a.gamma2, a.gamma3]


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


@tf.function
def train_step(model, optimizer, left, right, weights_list, w1, w2):
    with tf.GradientTape() as tape:
        outputs = model([left, right], training=True)
        loss, l_init, l_ref = model.compute_total_loss(
            left, right, outputs, weights_list, w1=w1, w2=w2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, l_init, l_ref, outputs


@tf.function
def eval_step(model, left, right, weights_list, w1, w2):
    outputs = model([left, right], training=False)
    loss, l_init, l_ref = model.compute_total_loss(
        left, right, outputs, weights_list, w1=w1, w2=w2)
    return loss, l_init, l_ref


def main():
    configure_gpu()

    train_ds, count = create_dataset(
        a.left_dir, a.right_dir, a.height, a.width, a.dataset, a.batch_size,
        shuffle=True, repeat=True)
    print('Num_data: {}'.format(count))

    val_ds = None
    val_count = 0
    if a.is_val:
        val_ds, val_count = create_dataset(
            a.left_val_dir, a.right_val_dir, a.height, a.width, a.dataset, a.batch_size,
            shuffle=False, repeat=True)
        print('Num_val: {}'.format(val_count))

    model = StereoNet(max_num_disparity=a.max_num_disparity)
    # Build once so variables exist before checkpoint restore.
    dummy_left = tf.zeros([a.batch_size, a.height, a.width, 3], dtype=tf.float32)
    dummy_right = tf.zeros([a.batch_size, a.height, a.width, 3], dtype=tf.float32)
    _ = model([dummy_left, dummy_right], training=True)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=a.lr,
        decay_steps=a.schedule_freq,
        decay_rate=0.5,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, a.checkpoint_dir, max_to_keep=8)

    if a.resume_dir is not None:
        latest = tf.train.latest_checkpoint(a.resume_dir)
        if latest is not None:
            ckpt.restore(latest).expect_partial()
            # Match original behavior: reset step counter after resume.
            optimizer.iterations.assign(0)
            print('Reload from: {}'.format(a.resume_dir))
        else:
            print('No checkpoint found in resume_dir; training from scratch')

    summary_writer = tf.summary.create_file_writer(a.summary_dir)
    train_iter = iter(train_ds)
    val_iter = iter(val_ds) if val_ds is not None else None

    for step in range(1, a.num_steps + 1):
        left, right = next(train_iter)
        loss, l_init, l_ref, outputs = train_step(
            model, optimizer, left, right, WEIGHTS_LIST, a.w1, a.w2)

        train_epoch = step * a.batch_size // count
        loss_v = float(loss)

        if step % a.summary_freq == 0:
            with summary_writer.as_default():
                tf.summary.scalar('learning_rate', optimizer.learning_rate(optimizer.iterations), step=step)
                tf.summary.scalar('step', step, step=step)
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar('loss_init', l_init, step=step)
                tf.summary.scalar('loss_ref', l_ref, step=step)
                tf.summary.image('left', deprocess(left), step=step, max_outputs=1)
                tf.summary.image('right', deprocess(right), step=step, max_outputs=1)
                tf.summary.image('left_disp_refined', outputs['left_refined_disp'], step=step, max_outputs=1)
                tf.summary.image('right_disp_refined', outputs['right_refined_disp'], step=step, max_outputs=1)
                tf.summary.image('left_disp_init', outputs['left_initial_disp'], step=step, max_outputs=1)
                tf.summary.image('right_disp_init', outputs['right_initial_disp'], step=step, max_outputs=1)
            print('-------- summary saved --------')

        if a.is_val and step % count == 0:
            print('Running Validation')
            total_vl = 0.0
            for _ in range(val_count):
                vleft, vright = next(val_iter)
                vl, _, _ = eval_step(
                    model, vleft, vright, WEIGHTS_LIST, a.w1, a.w2)
                total_vl += float(vl)
            vl_avg = total_vl / max(val_count, 1)
            with summary_writer.as_default():
                tf.summary.scalar('loss_val', vl_avg, step=step)
            print('-------- training_loss:{0:.4f}    validation_loss:{1:.4f}'.format(loss_v, vl_avg))

        if step % a.save_freq == 0:
            save_path = ckpt_manager.save(checkpoint_number=step)
            print('-------- checkpoint saved:{} --------'.format(save_path))

        if step % a.print_summary_freq == 0 or step == 1:
            print('epoch:{0}    step:{1}   loss:{2:.4f}'.format(train_epoch, step, loss_v))

    save_path = ckpt_manager.save(checkpoint_number=a.num_steps)
    print('-------- checkpoint saved:{} --------'.format(save_path))


if __name__ == "__main__":
    main()
