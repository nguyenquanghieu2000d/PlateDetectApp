from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.engine import training
from tensorflow.keras import layers
import tensorflow as tf
from src.setting import *
WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                       'keras-applications/vgg16/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

def Vgg16(weights='imagenet', input_shape=None):
    # Determine proper input shape
    # input_shape = imagenet_utils.obtain_input_shape(
    #     input_shape,
    #     default_size=224,
    #     min_size=32,
    #     data_format=backend.image_data_format(),
    #     require_flatten=False,
    #     weights=weights)
    #
    # img_input = layers.Input(shape=input_shape)
    #
    # # Block 1
    # x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    # x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    #
    # # Block 2
    # x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    #
    # # Block 3
    # x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    # x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    # x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    #
    # # Block 4
    # x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #
    # # Block 5
    # x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    #
    # inputs = img_input
    # # Create model.
    # model = training.Model(inputs, x, name='vgg16')

    model = tf.keras.models.Sequential()
    # Block 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    # # Block 4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    # Block 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Load weights.
    weights_path = data_utils.get_file(
        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path)

    return model
