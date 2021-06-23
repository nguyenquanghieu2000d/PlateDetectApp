import pandas as pd
import tensorflow as tf
import os
from src.setting import *

def get_pathframe(path):
    """
    Lấy toàn bộ image và nhãn tương ứng, xong đưa vào pandas dataframe
    """
    filenames = os.listdir(path)
    categories = []
    paths = []
    for filename in filenames:
        # paths.append(path+ filename)
        paths.append(os.path.join(path, filename))
        category = filename.split('.')[0]
        categories.append(CLASS_DIST[category])

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories,
        'paths': paths
    })
    return df


# df.tail(5)


def load_and_preprocess_image(path):
    """
    Load từng ảnh và thay đổi kích cỡ của chúng theo size model
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    image /= 255.0  # normalize to [0,1] range
    return image


def convert_to_tensor(df):
    """
    Chuyển đổi data về dạng tensor
    """
    path_ds = tf.data.Dataset.from_tensor_slices(df['paths'])
    image_ds = path_ds.map(load_and_preprocess_image)
    onehot_label = tf.one_hot(tf.cast(df['category'], tf.int64), len(CLASS))  # if using softmax
    label_ds = tf.data.Dataset.from_tensor_slices(onehot_label)

    return image_ds, label_ds
