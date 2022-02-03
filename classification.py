# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import pathlib


# 配置
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32


# 加载和格式化图片
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


def load_datasheet(path):
  # 检索图片
  data_root = pathlib.Path(path)

  all_image_paths = list(data_root.glob('*/*'))
  all_image_paths = [str(path) for path in all_image_paths]
  random.shuffle(all_image_paths)

  image_count = len(all_image_paths)

  # 列出可用的标签
  label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
  # 为每个标签分配索引
  label_to_index = dict((name, index) for index, name in enumerate(label_names))
  # 为每个标签分配索引
  all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                      for path in all_image_paths]

  # 构建一个 tf.data.Dataset
  # 将字符串数组切片，得到一个字符串数据集
  path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
  # 现在创建一个新的数据集，通过在路径数据集上映射 preprocess_image 来动态加载和格式化图片
  image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
  # 使用同样的 from_tensor_slices 方法你可以创建一个标签数据集
  label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
  # 由于这些数据集顺序相同，可以将他们打包在一起得到一个(图片, 标签)对数据集：
  # image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
  return (np.array(list(image_ds.as_numpy_iterator()), dtype=np.uint8), np.array(list(label_ds.as_numpy_iterator()), dtype=np.uint8))


# 导入数据集
(train_images, train_labels) = load_datasheet('./train')
(test_images, test_labels) = load_datasheet('./test')


# 构建模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(192, 192)),
    keras.layers.Dense(128, activation='relu'),
    # len(label_names)
    keras.layers.Dense(3)
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
# 向模型馈送数据
model.fit(train_images, train_labels, epochs=10)

# 评估准确率
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


# 保存模型
pathlib.Path('saved_model').mkdir()
model.save('saved_model/my_model')


# 进行预测
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
# 哪个标签的置信度值最大
print(np.argmax(predictions[0]))
# 检查测试标签
print(test_labels[0])


# 使用训练好的模型
img = test_images[1]
img = (np.expand_dims(img,0))
predictions_single = probability_model.predict(img)

print(predictions_single)
print(np.argmax(predictions_single[0]))

