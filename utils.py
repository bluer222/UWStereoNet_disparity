import tensorflow as tf


def deprocess(img):
    """Denormalize image from [-1, 1] to uint8 [0, 255]."""
    img = (img + 1) / 2
    return tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)


def lrelu(x, a=0.2):
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


class ConvNormLReLU(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=(1, 1), with_lrelu=True, **kwargs):
        super().__init__(**kwargs)
        self.with_lrelu = with_lrelu
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, strides=strides,
            padding='same', use_bias=False, name='conv')
        self.bn = tf.keras.layers.BatchNormalization(name='batch_norm')

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.with_lrelu:
            x = lrelu(x)
        return x


class DeconvNormLReLU(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=(2, 2), **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters, kernel_size=kernel_size, strides=strides,
            padding='same', use_bias=False, name='deconv')
        self.bn = tf.keras.layers.BatchNormalization(name='batch_norm')

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return lrelu(x)


class Conv3DNormLReLU(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=(1, 1, 1),
                 with_lrelu=True, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.with_lrelu = with_lrelu
        self.conv = tf.keras.layers.Conv3D(
            filters, kernel_size=kernel_size, strides=strides,
            padding=padding, use_bias=False, name='conv3d')
        self.bn = tf.keras.layers.BatchNormalization(name='batch_norm')

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.with_lrelu:
            x = lrelu(x)
        return x


class Deconv3DNormLReLU(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=(2, 2, 2), **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv3DTranspose(
            filters, kernel_size=kernel_size, strides=strides,
            padding='same', use_bias=False, name='deconv3d')
        self.bn = tf.keras.layers.BatchNormalization(name='batch_norm')

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return lrelu(x)


class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvNormLReLU(filters[0], strides=(1, 1), name='conv1')
        self.conv2 = ConvNormLReLU(filters[1], strides=(1, 1), name='conv2')
        self.conv3 = ConvNormLReLU(filters[2], strides=(1, 1), with_lrelu=False, name='conv3')

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return lrelu(inputs + x)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvNormLReLU(filters[0], strides=strides, name='conv1')
        self.conv2 = ConvNormLReLU(filters[1], strides=(1, 1), name='conv2')
        self.conv3 = ConvNormLReLU(filters[2], strides=(1, 1), with_lrelu=False, name='conv3')
        self.shortcut = ConvNormLReLU(filters[2], strides=strides, with_lrelu=False, name='shortcut')

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        shortcut = self.shortcut(inputs, training=training)
        return lrelu(shortcut + x)
