from tensorflow.keras import layers
from src.VGG16Implement import Vgg16
from src.dataPreprocessing import *

from src.setting import *

datasetPath = "./dataset/dataset/"

dataset_train = ""
dataset_test = ""
for i in os.listdir(datasetPath):
    df1 = get_pathframe(os.path.join(datasetPath,i))
    X1, Y1 = convert_to_tensor(df1)
    dataset1 = tf.data.Dataset.zip((X1, Y1)).shuffle(buffer_size=6500)
# dataset2 = tf.data.Dataset.zip((X2, Y2)).shuffle(buffer_size=6500)
# dataset3 = tf.data.Dataset.zip((X3, Y3)).shuffle(buffer_size=6500)
    dataset_train1 = dataset1.take(5000)
    dataset_test1 = dataset1.skip(5000)
# dataset_train2 = dataset2.take(5000)
# dataset_test2 = dataset2.skip(5000)
# dataset_train3 = dataset3.take(5000)
# dataset_test3 = dataset3.skip(5000)
    if dataset_train == "":
        dataset_train = dataset_train1
    else:
        dataset_train = dataset_train.concatenate(dataset_train1)

    if dataset_test == "":
        dataset_test = dataset_test1
    else:
        dataset_test = dataset_test.concatenate(dataset_test1)

# dataset_test = dataset_test1.concatenate(dataset_test2).concatenate(dataset_test3)
dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=True)


def VGGmodel():
    """
    Sử dụng vgg16 theo cách transfer learning
    :return: model nhận dạng có dạng save_model.pb
    """
    pre_trained_model = Vgg16(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # for layer in pre_trained_model.layers[:15]:
    #     layer.trainable = False
    # for layer in pre_trained_model.layers[15:]:
    #     layer.trainable = True
    last_layer = pre_trained_model.get_layer('block5_pool')
    last_output = last_layer.output
    x = layers.Flatten()(last_output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(len(CLASS), activation='softmax')(x)
    vggmodel = tf.keras.models.Model(pre_trained_model.input, x)

    vggmodel.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.SGD(lr=1e-4),
                     metrics=['accuracy'])

    return vggmodel


model = VGGmodel()

# print("OPKO")
model.summary()
hist1 = model.fit(dataset_train, epochs=5, validation_data=dataset_test)
model.save("./models/my_model")
