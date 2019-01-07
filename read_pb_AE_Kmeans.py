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
model_filename = r"model_saver\AE_Kmeans\pb_model.pb"

# pic_path = r'E:\dataset\For_super_resolution\test'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_ori = mnist.test.images
x_ori = np.reshape(x_ori,(-1,28,28,1))

print("x_ori shape = ", x_ori.shape)
print('x_ori dtype = ', x_ori.dtype)
# print('x_ori content = ', x_ori[0])
# print("x_noise shape = ",x_noise.shape)
# print('x_noise dtype = ',x_noise.dtype)
# print('x_noise content = ',x_noise[0])


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
    encode2 = sess.graph.get_tensor_by_name("pool2:0")
    print(result.shape)

    test_num = 21
    reconstruct = sess.run(result, feed_dict={ori_x: x_ori[test_num:test_num+1]})
    encode_data = sess.run(encode2,feed_dict={ori_x: x_ori[test_num:test_num+1]})
    print("reconstruct.shape = ",reconstruct.shape)
    print("encode_data.shape = ", encode_data.shape)



    '''
    display plots
    '''
    plt.figure()
    plt.subplot(2,1,1)
    x_show = np.reshape(x_ori[test_num],(28,28))
    plt.imshow(x_show)

    reconstruct = np.reshape(reconstruct,(-1,28,28))
    plt.subplot(2,1,2)
    plt.imshow(reconstruct[0])

    plt.show()