import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import skimage
import common as cm
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.examples.tutorials.mnist.input_data as input_data

# model_filename = r"model_saver\AE_mnist\pb_model.pb"
model_filename = r"model_saver\AE_circle\pb_model.pb"


pic_path = r'E:\dataset\Surface_detection'
(input_train, input_train_label, no_care, no_care_2) = cm.data_load(pic_path, train_ratio=1, resize=(64, 64),
                                                                    shuffle=True, normalize=True, has_dir=True)

print('x_train shape = ', input_train.shape)
print('x_train_label shape = ', input_train_label.shape)
# print(input_train_label)


# 設定GPU參數
config = tf.ConfigProto(log_device_placement=True,
                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                        )
config.gpu_options.per_process_gpu_memory_fraction = 0.1
with tf.Session(config=config) as sess:
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()

        tf.import_graph_def(graph_def, name='')  # 導入計算圖

    sess.run(tf.global_variables_initializer())

    ori_x = sess.graph.get_tensor_by_name("input_x:0")
    #supressed_x = sess.graph.get_tensor_by_name("supressed_x:0")
    # print(input_x.shape)
    result = sess.graph.get_tensor_by_name("output/Relu:0")
    encode4 = sess.graph.get_tensor_by_name("pool4:0")
    print(result.shape)

    test_num = 22
    reconstruct = sess.run(result, feed_dict={ori_x: input_train[test_num:test_num+1]})
    encode_data = sess.run(encode4,feed_dict={ori_x: input_train[test_num:test_num+1]})
    print("reconstruct.shape = ",reconstruct.shape)
    print("encode_data.shape = ", encode_data.shape)



'''
display plots
'''
plt.figure()
plt.subplot(2,1,1)
x_show = input_train[test_num]
print(x_show.shape)
plt.imshow(x_show)

#reconstruct = np.reshape(reconstruct,(-1,28,28))
plt.subplot(2,1,2)
plt.imshow(reconstruct[0])

plt.show()