# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import matplotlib.pyplot as plt
import numpy as np
import random
import pathlib


# 配置
AUTOTUNE = tf.data.experimental.AUTOTUNE
#  32: 0.514
#  64: 0.532 (1*OOM)
# 128: 0.5228 (2*OOM)
BATCH_SIZE = 32
# (96, 128, 160, 192, 224)
# 192: 0.513; 224: 0.531
IMAGE_SIZE = 224


# 加载和格式化图片
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
  image /= 255.0  # normalize to [0,1] range
  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


def load_datasheet(path, disp_one=False):
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
  label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int8))
  # 由于这些数据集顺序相同，可以将他们打包在一起得到一个(图片, 标签)对数据集：
  image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

  if disp_one:
    img_path = all_image_paths[0]
    label = all_image_labels[0]

    plt.imshow(load_and_preprocess_image(img_path))
    plt.grid(False)
    plt.xlabel(pathlib.Path(img_path).relative_to(data_root))
    plt.title(label_names[label].title())
    plt.show()

  return image_label_ds, image_count


def training():
  # 导入数据集
  image_label_ds, image_count = load_datasheet('./train')

  # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
  # 被充分打乱。
  ds = image_label_ds.shuffle(buffer_size=5000)
  ds = ds.repeat()
  ds = ds.batch(BATCH_SIZE)
  # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
  ds = ds.prefetch(buffer_size=AUTOTUNE)


  # 构建模型
  mobile_net = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False)
  # 设置 MobileNet 的权重为不可训练
  mobile_net.trainable=False

  # 在将输出传递给 MobilNet 模型之前，需要将其范围从 [0,1] 转化为 [-1,1]：
  def change_range(image,label):
    return 2*image-1, label

  keras_ds = ds.map(change_range)

  # # 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
  model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    # len(label_names)
    tf.keras.layers.Dense(3, activation = 'softmax')])

  # 编译模型
  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"])


  steps_per_epoch=tf.math.ceil(image_count/BATCH_SIZE).numpy()

  # 训练模型
  # 向模型馈送数据
  model.fit(ds, epochs=10, steps_per_epoch=steps_per_epoch)


  # 导入数据集
  test_ds, test_count = load_datasheet('./test')

  test_ds = test_ds.batch(BATCH_SIZE)
  # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
  test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

  # 评估准确率
  test_loss, test_acc = model.evaluate(test_ds, verbose=2)
  print('\nTest accuracy:', test_acc)


  # 保存模型
  pathlib.Path('saved_model').mkdir()
  model.save('saved_model/my_model')


class_names = sorted(item.name for item in pathlib.Path('./test').glob('*/') if item.is_dir())
CLASS_NUM = len(class_names)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(CLASS_NUM))
  plt.yticks([])
  thisplot = plt.bar(range(CLASS_NUM), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


def prob():
  # 从保存的模型重新加载一个新的 Keras 模型
  model = tf.keras.models.load_model('saved_model/my_model')

  # 导入数据集
  test_ds, test_count = load_datasheet('./test')

  test_ds = test_ds.batch(BATCH_SIZE)
  # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
  test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

  # 进行预测
  probability_model = tf.keras.Sequential([model,
                                          tf.keras.layers.Softmax()])
  predictions = probability_model.predict(test_ds)
  print(predictions[0])
  # 哪个标签的置信度值最大
  print(np.argmax(predictions[0]))
  # 检查测试标签
  test_list = list(test_ds)
  test_images = np.array(test_list[0][0])
  test_labels = np.array(test_list[0][1])
  print(test_labels[0])


  # Plot the first X test images, their predicted labels, and the true labels.
  # Color correct predictions in blue and incorrect predictions in red.
  num_rows = 5
  num_cols = 3
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
  plt.tight_layout()
  plt.show()

  # 使用训练好的模型
  img = test_images[1]
  img = (np.expand_dims(img,0))
  predictions_single = probability_model.predict(img)

  print(predictions_single)
  print(np.argmax(predictions_single[0]))
  print(test_labels[1])
  plot_value_array(1, predictions_single[0], test_labels)
  _ = plt.xticks(range(CLASS_NUM), class_names, rotation=45)


if __name__ == "__main__":
  training()
