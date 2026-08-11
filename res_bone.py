import tensorflow as tf
from utils import ConvNormLReLU, IdentityBlock, ConvBlock


class ContextBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale_1 = ConvNormLReLU(32, kernel_size=1, name='scale_1')
        self.scale_2 = ConvNormLReLU(32, kernel_size=1, name='scale_2')
        self.scale_3 = ConvNormLReLU(32, kernel_size=1, name='scale_3')
        self.output_conv = tf.keras.layers.Conv2D(128, 1, padding='same', use_bias=True, name='output')

    def call(self, inputs, training=None):
        size = tf.shape(inputs)[1:3]

        s1 = tf.nn.avg_pool2d(inputs, ksize=2, strides=2, padding='VALID')
        s1 = self.scale_1(s1, training=training)
        s1 = tf.image.resize(s1, size)

        s2 = tf.nn.avg_pool2d(inputs, ksize=4, strides=4, padding='VALID')
        s2 = self.scale_2(s2, training=training)
        s2 = tf.image.resize(s2, size)

        s3 = tf.nn.avg_pool2d(inputs, ksize=8, strides=8, padding='VALID')
        s3 = self.scale_3(s3, training=training)
        s3 = tf.image.resize(s3, size)

        concat = tf.concat([inputs, s1, s2, s3], axis=-1)
        return self.output_conv(concat)


class ResBone(tf.keras.layers.Layer):
    """Shared stereo backbone producing disparity features and segmentation embedding."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvNormLReLU(32, kernel_size=7, strides=(2, 2), with_lrelu=False, name='conv1')

        self.l2a = ConvBlock([32, 32, 64], strides=(1, 1), name='l2a')
        self.l2b = IdentityBlock([32, 32, 64], name='l2b')
        self.l2c = IdentityBlock([32, 32, 64], name='l2c')

        self.l3a = ConvBlock([64, 64, 64], strides=(2, 2), name='l3a')
        self.l3b = IdentityBlock([64, 64, 64], name='l3b')
        self.l3c = IdentityBlock([64, 64, 64], name='l3c')
        self.l3d = IdentityBlock([64, 64, 64], name='l3d')

        self.l4a = ConvBlock([64, 64, 128], strides=(2, 2), name='l4a')
        self.l4b = IdentityBlock([64, 64, 128], name='l4b')
        self.l4c = IdentityBlock([64, 64, 128], name='l4c')

        self.context = ContextBlock(name='context_block')

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)

        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)

        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        disp_feature = self.l3d(x, training=training)

        x = self.l4a(disp_feature, training=training)
        x = self.l4b(x, training=training)
        x = self.l4c(x, training=training)
        seg_embedding = self.context(x, training=training)

        return disp_feature, seg_embedding


# Backward-compatible alias matching the original class name.
Res_bone = ResBone
