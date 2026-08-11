import tensorflow as tf
from utils import (
    Conv3DNormLReLU, Deconv3DNormLReLU, ConvNormLReLU, DeconvNormLReLU,
    ConvBlock, IdentityBlock,
)

SCALE = 1
THRESHOLD = 10


def left_concatenate_features(left, right, max_num_disparity):
    """Build left-referenced cost volume [n, d, h, w, 2f]."""
    width = tf.shape(left)[2]
    max_num_features = max_num_disparity // 4
    concated_features = []
    for d in range(max_num_features + 1):
        right_features = right[:, :, :width - d, :]
        right_features = tf.pad(right_features, [[0, 0], [0, 0], [d, 0], [0, 0]])
        features = tf.concat([left, right_features], axis=-1)
        concated_features.append(features)
    return tf.stack(concated_features, axis=1)


def right_concatenate_features(right, left, max_num_disparity):
    """Build right-referenced cost volume [n, d, h, w, 2f]."""
    max_num_features = max_num_disparity // 4
    concated_features = []
    for d in range(max_num_features + 1):
        left_features = left[:, :, d:, :]
        left_features = tf.pad(left_features, [[0, 0], [0, 0], [0, d], [0, 0]])
        features = tf.concat([right, left_features], axis=-1)
        concated_features.append(features)
    return tf.stack(concated_features, axis=1)


def create_costVolume(left_res, right_res, max_num_disparity):
    left_cost = left_concatenate_features(left_res, right_res, max_num_disparity)
    right_cost = right_concatenate_features(right_res, left_res, max_num_disparity)
    return left_cost, right_cost


class ResShortcut(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv3DNormLReLU(filters, 3, strides=(1, 1, 1), name='conv1')
        self.conv2 = Conv3DNormLReLU(filters, 3, strides=(1, 1, 1), name='conv2')

    def call(self, x, y, training=None):
        y = self.conv1(y, training=training)
        y = self.conv2(y, training=training)
        return x + y


class Module3D(tf.keras.layers.Layer):
    """3D hourglass over the cost volume. Input [n,d,h,w,f*2] -> [n,h,w,d]."""

    def __init__(self, num_features, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.conv = Conv3DNormLReLU(num_features, 5, strides=(1, 1, 1), padding='valid', name='3Dconv')
        self.conv0 = Conv3DNormLReLU(num_features, 3, strides=(2, 2, 2), name='3Dconv0')
        self.conv1 = Conv3DNormLReLU(num_features, 3, strides=(2, 2, 2), name='3Dconv1')
        self.conv2 = Conv3DNormLReLU(num_features, 3, strides=(2, 2, 2), name='3Dconv2')
        self.conv3 = Conv3DNormLReLU(num_features, 3, strides=(2, 2, 2), name='3Dconv3')

        self.deconv4 = Deconv3DNormLReLU(num_features, 3, strides=(2, 2, 2), name='3Ddeconv4')
        self.shortcut1 = ResShortcut(num_features, name='shortcut1')
        self.deconv5 = Deconv3DNormLReLU(num_features, 3, strides=(2, 2, 2), name='3Ddeconv5')
        self.shortcut2 = ResShortcut(num_features, name='shortcut2')
        self.deconv6 = Deconv3DNormLReLU(num_features, 3, strides=(2, 2, 2), name='3Ddeconv6')
        self.shortcut3 = ResShortcut(num_features, name='shortcut3')
        self.deconv7 = Deconv3DNormLReLU(num_features, 3, strides=(2, 2, 2), name='3Ddeconv7')
        self.shortcut4 = ResShortcut(num_features, name='shortcut4')
        self.out_conv = tf.keras.layers.Conv3D(1, 3, strides=(1, 1, 1), padding='same', use_bias=False, name='conv3d')

    def call(self, inputs, training=None):
        padded = tf.pad(inputs, [[0, 0], [1, 2], [2, 2], [2, 2], [0, 0]])
        conv = self.conv(padded, training=training)
        conv0 = self.conv0(conv, training=training)
        conv1 = self.conv1(conv0, training=training)
        conv2 = self.conv2(conv1, training=training)
        conv3 = self.conv3(conv2, training=training)

        x = self.deconv4(conv3, training=training)
        x = self.shortcut1(x, conv2, training=training)
        x = self.deconv5(x, training=training)
        x = self.shortcut2(x, conv1, training=training)
        x = self.deconv6(x, training=training)
        x = self.shortcut3(x, conv0, training=training)
        x = self.deconv7(x, training=training)
        x = self.shortcut4(x, conv, training=training)

        x = self.out_conv(x)
        x = tf.squeeze(x, axis=-1)
        return tf.transpose(x, [0, 2, 3, 1])


class Refinement(tf.keras.layers.Layer):
    def __init__(self, disparity_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvNormLReLU(64, kernel_size=5, strides=(2, 2), with_lrelu=False, name='conv1')
        self.l2a = ConvBlock([32, 32, 64], strides=(1, 1), name='l2a')
        self.l2b = IdentityBlock([32, 32, 64], name='l2b')
        self.l2c = IdentityBlock([32, 32, 64], name='l2c')
        self.l3a = ConvBlock([64, 64, 128], strides=(1, 1), name='l3a')
        self.l3b = IdentityBlock([64, 64, 128], name='l3b')
        self.l3c = IdentityBlock([64, 64, 128], name='l3c')
        self.deconv = DeconvNormLReLU(64, 3, strides=(2, 2), name='deconvl4a')
        self.output_conv = tf.keras.layers.Conv2D(disparity_channels, 3, padding='same', name='output')

    def call(self, disp_logits, seg_embedding, training=None):
        seg_embedding = tf.image.resize(seg_embedding, tf.shape(disp_logits)[1:3])
        x = tf.concat([disp_logits, seg_embedding], axis=-1)
        x = self.conv1(x, training=training)
        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)
        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        x = self.deconv(x, training=training)
        residual = self.output_conv(x)
        return disp_logits + residual


def reconstruct_img(inputs, disp):
    """Bilinear sampler (monodepth). disp shifts source x coordinates."""
    num_batch = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    num_channels = tf.shape(inputs)[3]
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)

    def _repeat(x, n_repeats):
        rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
        return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        x = tf.clip_by_value(x, 0.0, width_f - 1)
        x0_f = tf.floor(x)
        y0_f = tf.floor(y)
        x1_f = x0_f + 1
        x0 = tf.cast(x0_f, tf.int32)
        y0 = tf.cast(y0_f, tf.int32)
        x1 = tf.cast(tf.minimum(x1_f, width_f - 1), tf.int32)

        dim2 = width
        dim1 = width * height
        base = _repeat(tf.range(num_batch) * dim1, height * width)
        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_flat = tf.reshape(im, tf.stack([-1, num_channels]))
        pix_l = tf.gather(im_flat, idx_l)
        pix_r = tf.gather(im_flat, idx_r)
        weight_l = tf.expand_dims(x1_f - x, 1)
        weight_r = tf.expand_dims(x - x0_f, 1)
        return weight_l * pix_l + weight_r * pix_r

    x_t, y_t = tf.meshgrid(
        tf.linspace(0.0, width_f - 1.0, width),
        tf.linspace(0.0, height_f - 1.0, height))
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    x_t_flat = tf.tile(x_t_flat, tf.stack([num_batch, 1]))
    y_t_flat = tf.tile(y_t_flat, tf.stack([num_batch, 1]))
    x_t_flat = tf.reshape(x_t_flat, [-1]) + tf.reshape(disp, [-1])
    y_t_flat = tf.reshape(y_t_flat, [-1])

    sampled = _interpolate(inputs, x_t_flat, y_t_flat)
    return tf.reshape(sampled, tf.stack([num_batch, height, width, num_channels]))


def reconstruct_right(inputs, disp):
    return reconstruct_img(inputs, disp)


def reconstruct_left(inputs, disp):
    return reconstruct_img(inputs, -disp)


def resize_tensor(inputs, shape):
    """Bilinear upsample logits to [H, W, F]."""
    inputs = tf.image.resize(inputs, [shape[0], shape[1]])
    inputs = tf.transpose(inputs, [0, 1, 3, 2])
    inputs = tf.image.resize(inputs, [shape[0], shape[2]])
    return tf.transpose(inputs, [0, 1, 3, 2])


def predict(inputs, target_shape):
    """Soft-argmax disparity prediction [bz, h, w, 1]."""
    inputs = resize_tensor(inputs, target_shape)
    score = tf.nn.softmax(inputs * SCALE, axis=-1)
    ind = tf.range(target_shape[-1], dtype=tf.float32)
    ind = tf.reshape(ind, [1, 1, 1, -1])
    score = tf.reduce_sum(score * ind, axis=-1, keepdims=True)
    return score


def gradient(inputs, axis):
    if axis == 'x':
        inputs = tf.pad(inputs, [[0, 0], [0, 0], [1, 0], [0, 0]], 'SYMMETRIC')
        return inputs[:, :, 1:, :] - inputs[:, :, :-1, :]
    if axis == 'y':
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0], [0, 0]], 'SYMMETRIC')
        return inputs[:, 1:, :, :] - inputs[:, :-1, :, :]
    raise ValueError('axis should be either x or y')


def compute_SSIM_loss(x, y):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = tf.nn.avg_pool2d(x, ksize=3, strides=1, padding='VALID')
    mu_y = tf.nn.avg_pool2d(y, ksize=3, strides=1, padding='VALID')
    sigma_x = tf.nn.avg_pool2d(x ** 2, ksize=3, strides=1, padding='VALID') - mu_x ** 2
    sigma_y = tf.nn.avg_pool2d(y ** 2, ksize=3, strides=1, padding='VALID') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool2d(x * y, ksize=3, strides=1, padding='VALID') - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = (1 - ssim_n / ssim_d) / 2
    return tf.reduce_mean(ssim)


def unary_term(left, right, recons_left, recons_right):
    left = left[:, :, THRESHOLD:, :]
    recons_left = recons_left[:, :, THRESHOLD:, :]
    right = right[:, :, :-THRESHOLD, :]
    recons_right = recons_right[:, :, :-THRESHOLD, :]

    ssim_loss = compute_SSIM_loss(recons_left, left) + compute_SSIM_loss(recons_right, right)
    l_img = tf.reduce_mean(tf.abs(left - recons_left))
    r_img = tf.reduce_mean(tf.abs(right - recons_right))

    left_grad_loss = (
        tf.reduce_mean(tf.abs(gradient(left, 'x') - gradient(recons_left, 'x'))) +
        tf.reduce_mean(tf.abs(gradient(left, 'y') - gradient(recons_left, 'y'))))
    right_grad_loss = (
        tf.reduce_mean(tf.abs(gradient(right, 'x') - gradient(recons_right, 'x'))) +
        tf.reduce_mean(tf.abs(gradient(right, 'y') - gradient(recons_right, 'y'))))

    return 0.85 * ssim_loss + 0.15 * (l_img + r_img) + 0.15 * (left_grad_loss + right_grad_loss)


def regularization_term(disp, img):
    disp_gx = tf.abs(disp[:, :, 2:, :] + disp[:, :, :-2, :] - 2 * disp[:, :, 1:-1, :])
    disp_gy = tf.abs(disp[:, 2:, :, :] + disp[:, :-2, :, :] - 2 * disp[:, 1:-1, :, :])
    img_gx = tf.abs(img[:, :, 2:, :] + img[:, :, :-2, :] - 2 * img[:, :, 1:-1, :])
    img_gy = tf.abs(img[:, 2:, :, :] + img[:, :-2, :, :] - 2 * img[:, 1:-1, :, :])
    weights_x = tf.exp(-tf.reduce_mean(img_gx, 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(img_gy, 3, keepdims=True))
    return tf.reduce_mean(disp_gx * weights_x) + tf.reduce_mean(disp_gy * weights_y)


def smooth_term(disp, seg_embedding, diff):
    disp_gx = tf.abs(gradient(disp, 'x'))
    disp_gy = tf.abs(gradient(disp, 'y'))
    emb_gx = tf.abs(gradient(seg_embedding, 'x'))
    emb_gy = tf.abs(gradient(seg_embedding, 'y'))
    mean_x = tf.reduce_mean(emb_gx, 3, keepdims=True)
    mean_y = tf.reduce_mean(emb_gy, 3, keepdims=True)
    diff = tf.clip_by_value(diff, 0, 3)
    weights_x = tf.exp(-(3 - diff)) + tf.exp(-mean_x * 5)
    weights_y = tf.exp(-(3 - diff)) + tf.exp(-mean_y * 5)
    return tf.reduce_mean(disp_gx * weights_x) + tf.reduce_mean(disp_gy * weights_y)


def consistency_term(img, recons_img, is_left):
    if is_left:
        return tf.reduce_mean(tf.abs(img[:, :, THRESHOLD:, :] - recons_img[:, :, THRESHOLD:, :]))
    return tf.reduce_mean(tf.abs(img[:, :, :-THRESHOLD, :] - recons_img[:, :, :-THRESHOLD, :]))


def compute_loss(left, right, left_disp, right_disp, left_embedding, right_embedding, weights_list, name):
    recons_right = reconstruct_right(left, right_disp)
    recons_left = reconstruct_left(right, left_disp)
    re_recons_right = reconstruct_right(recons_left, right_disp)
    re_recons_left = reconstruct_left(recons_right, left_disp)

    l_consis = consistency_term(left, re_recons_left, True) + consistency_term(right, re_recons_right, False)

    if name == 'Initial_loss':
        l_unary = unary_term(left, right, recons_left, recons_right)
        l_reg = regularization_term(left_disp, left) + regularization_term(right_disp, right)
        return weights_list[0] * l_unary + weights_list[1] * l_reg + weights_list[2] * l_consis

    left_embedding = tf.image.resize(left_embedding, tf.shape(left_disp)[1:3])
    right_embedding = tf.image.resize(right_embedding, tf.shape(right_disp)[1:3])
    recons_right_disp = reconstruct_right(left_disp, right_disp)
    recons_left_disp = reconstruct_left(right_disp, left_disp)
    left_diff = tf.abs(left_disp - recons_left_disp)
    right_diff = tf.abs(right_disp - recons_right_disp)
    l_unary = unary_term(left, right, recons_left, recons_right)
    l_smooth = (
        smooth_term(left_disp, left_embedding, left_diff) +
        smooth_term(right_disp, right_embedding, right_diff))
    return weights_list[3] * l_unary + weights_list[4] * l_smooth + weights_list[5] * l_consis


class StereoNet(tf.keras.Model):
    """UWStereoNet disparity estimation model."""

    def __init__(self, max_num_disparity=192, feature_channels=64, **kwargs):
        super().__init__(**kwargs)
        from res_bone import ResBone as _ResBone
        self.max_num_disparity = max_num_disparity
        self.backbone = _ResBone(name='Res_bone')
        self.module3d = Module3D(feature_channels, name='modual3D')
        # Cost volume has max_disp/4+1 disparity samples; the valid 5^3 conv
        # in Module3D reduces that by 1, matching the original TF1 shapes.
        disp_channels = max_num_disparity // 4
        self.refinement = Refinement(disp_channels, name='refinement')

    def call(self, inputs, training=None):
        left, right = inputs
        left_feat, left_emb = self.backbone(left, training=training)
        right_feat, right_emb = self.backbone(right, training=training)

        left_cv, right_cv = create_costVolume(left_feat, right_feat, self.max_num_disparity)

        left_init_logits = self.module3d(left_cv, training=training)
        right_init_logits = self.module3d(right_cv, training=training)

        h = tf.shape(left)[1]
        w = tf.shape(left)[2]
        target_shape = [h, w, self.max_num_disparity + 1]

        left_init_disp = predict(left_init_logits, target_shape)
        right_init_disp = predict(right_init_logits, target_shape)

        left_ref_logits = self.refinement(left_init_logits, left_emb, training=training)
        right_ref_logits = self.refinement(right_init_logits, right_emb, training=training)
        left_ref_disp = predict(left_ref_logits, target_shape)
        right_ref_disp = predict(right_ref_logits, target_shape)

        return {
            'left_initial_disp': left_init_disp,
            'right_initial_disp': right_init_disp,
            'left_refined_disp': left_ref_disp,
            'right_refined_disp': right_ref_disp,
            'left_embedding': left_emb,
            'right_embedding': right_emb,
        }

    def compute_total_loss(self, left, right, outputs, weights_list, w1=0.3, w2=0.7):
        l_init = compute_loss(
            left, right,
            outputs['left_initial_disp'], outputs['right_initial_disp'],
            outputs['left_embedding'], outputs['right_embedding'],
            weights_list, name='Initial_loss')
        l_ref = compute_loss(
            left, right,
            outputs['left_refined_disp'], outputs['right_refined_disp'],
            outputs['left_embedding'], outputs['right_embedding'],
            weights_list, name='Refined_loss')
        return w1 * l_init + w2 * l_ref, l_init, l_ref
