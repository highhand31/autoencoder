import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import common as cm

#Parameters
save_path = r"model_saver\AE_pill"
out_dir_prefix=os.path.join(save_path,"model")
height = 300
width = 428
epochs = 50
GPU_ratio = 0.2
batch_size = 12

pic_path = r'E:\dataset\Halcon_Pill\Original\good'

files = [file.path for file in os.scandir(pic_path) if file.is_file()]

x_train = []

for file in files:
    img = cv2.imread(file)

    # print(img.shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (width, height))

    # flatten_num = img.shape[0]*img.shape[1]*img.shape[2]

    # img = img.reshape(flatten_num)

    x_train.append(img)

x_train = np.array(x_train)

print(x_train.shape)

pic_path = r'E:\dataset\Halcon_Pill\test'

files = [file.path for file in os.scandir(pic_path) if file.is_file()]

x_test = []

for file in files:
    img = cv2.imread(file)

    # print(img.shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (width, height))

    #     flatten_num = img.shape[0]*img.shape[1]*img.shape[2]

    #     img = img.reshape(flatten_num)

    x_test.append(img)

x_test = np.array(x_test)

print(x_test.shape)

# 格式轉換

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# 特徵縮放
x_train = x_train / 255
x_test = x_test / 255

train_ratio = 0.7
pic_path = r'E:\dataset\Halcon_Pill\Original'
(x_train_2,x_train_label_2,x_test_2,x_test_label_2) = data_load(pic_path,train_ratio,resize=(width,height))

print('x_train shape = ',x_train_2.shape)
print('x_train_label shape = ',x_train_label_2.shape)

print('x_test shape = ',x_test_2.shape)
print('x_test_label shape = ',x_test_label_2.shape)

x_test_2[0]

input_x = tf.placeholder(tf.float32,[None,height,width,3],name = "input_x")

#inference
kel_x = 9
kel_y = 9
with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    reuse=tf.AUTO_REUSE):
    net1 = slim.conv2d(input_x, 6, [kel_x, 1], scope='encode1_1')
    net2 = slim.conv2d(input_x, 6, [1, kel_y], scope='encode1_2')
    net3 = slim.conv2d(input_x, 4, [kel_x, kel_y], scope='encode1_3')
    # print("net1 shape = ",net1.shape)
    net = tf.concat([net1, net2, net3], 3)
    # net = slim.conv2d(net, 8, [9, 9], scope='encode1_2')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    print("encode1 shape = ", net.shape)

    # net shape = 32 x 32 x 16

    #     net = slim.conv2d(net, 16, [7, 7], scope='encode2')
    net1 = slim.conv2d(net, 12, [kel_x, 1], scope='encode2_1')
    net2 = slim.conv2d(net, 12, [1, kel_y], scope='encode2_2')
    net3 = slim.conv2d(net, 8, [kel_x, kel_y], scope='encode2_3')
    net = tf.concat([net1, net2, net3], 3)
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    print("encode2 shape = ", net.shape)

    # net shape = 16 x 16 x 32

    #     net = slim.conv2d(net, 32, [5, 5], scope='encode3')
    net1 = slim.conv2d(net, 24, [kel_x, 1], scope='encode3_1')
    net2 = slim.conv2d(net, 24, [1, kel_y], scope='encode3_2')
    net3 = slim.conv2d(net, 16, [kel_x, kel_y], scope='encode3_3')
    net = tf.concat([net1, net2, net3], 3)
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    print("encode3 shape = ", net.shape)

    # net shape = 8 x 8 x 64

    #     net = slim.conv2d(net, 64, [3, 3], scope='encode4')
    net1 = slim.conv2d(net, 48, [kel_x, 1], scope='encode4_1')
    net2 = slim.conv2d(net, 48, [1, kel_y], scope='encode4_2')
    net3 = slim.conv2d(net, 32, [kel_x, kel_y], scope='encode4_3')
    net = tf.concat([net1, net2, net3], 3)
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    print("encode4 shape = ", net.shape)

    # net shape = 4 x 4 x 128

    net = slim.conv2d_transpose(net, 32, [3, 3], stride=2, activation_fn=None, padding="VALID", scope="unpool1")
    # net = unpooling(net,([16,16]))
    net1 = slim.conv2d(net, 24, [kel_x, 1], scope='decode1_1')
    net2 = slim.conv2d(net, 24, [1, kel_y], scope='decode1_2')
    net3 = slim.conv2d(net, 16, [kel_x, kel_y], scope='decode1_3')
    net = tf.concat([net1, net2, net3], 3)
    # net = slim.conv2d(net, 32, [3, 3], scope='decode1')

    print("decode1 shape = ", net.shape)

    # net shape = 8 x 8 x 32

    net = slim.conv2d_transpose(net, 16, [3, 3], stride=2, padding="VALID", scope="unpool2")
    # net = unpooling(net,[32,32])
    # net = slim.conv2d(net, 16, [5, 5], scope='decode2')
    net1 = slim.conv2d(net, 12, [kel_x, 1], scope='decode2_1')
    net2 = slim.conv2d(net, 12, [1, kel_y], scope='decode2_2')
    net3 = slim.conv2d(net, 8, [kel_x, kel_y], scope='decode2_3')
    net = tf.concat([net1, net2, net3], 3)

    print("decode2 shape = ", net.shape)

    # net shape = 16 x 16 x 16

    net = slim.conv2d_transpose(net, 8, [2, 2], stride=2, padding="VALID", scope="unpool3")
    # net = unpooling(net,[64,64])
    # net = slim.conv2d(net, 8, [7, 7], scope='decode3')
    net1 = slim.conv2d(net, 6, [kel_x, 1], scope='decode3_1')
    net2 = slim.conv2d(net, 6, [1, kel_y], scope='decode3_2')
    net3 = slim.conv2d(net, 4, [kel_x, kel_y], scope='decode3_3')
    net = tf.concat([net1, net2, net3], 3)

    print("decode3 shape = ", net.shape)

    # net shape = 32 x 32 x 8

    net = slim.conv2d_transpose(net, 3, [2, 2], stride=2, padding="VALID", scope="unpool4")
    # net = unpooling(net,[64,64])
    # net = slim.conv2d(net, 3, [9, 9],activation_fn=tf.nn.sigmoid,scope='decode4')
    net1 = slim.conv2d(net, 1, [kel_x, 1], scope='decode4_1')
    net2 = slim.conv2d(net, 2, [1, kel_y], scope='decode4_2')
    net = tf.concat([net1, net2], 3)

    print("decode4 shape = ", net.shape)

    # net shape = 64 x 64 x 3

#optimizer
loss = tf.reduce_mean(tf.pow(net-input_x,2))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


def optimizer_GD(dataInput, batch_choice=50):
    data_length = dataInput.shape[0]

    total_batch = data_length // batch_choice

    vali_loss = 0;
    vali_acc = 0

    # vali_x_temp=[];vali_y_temp=[]

    loss = 0;
    acc = 0;

    if (data_length % batch_choice):
        total_batch += 1

    for i in range(total_batch):

        num_start = i * batch_choice

        num_end = num_start + batch_choice

        if (num_end >= data_length) and batch_choice > 1:
            num_end = data_length - 1

        # print('i=',i)

        # print('num_start',num_start)

        # print('num_end',num_end)

        sess.run(optimizer, feed_dict={ \
            input_x: dataInput[num_start:num_end]})


#set up model saver
if not os.path.exists(save_path):
    os.makedirs(save_path)

saver = tf.train.Saver(max_to_keep=20)


#training
# 設定GPU參數
config = tf.ConfigProto(log_device_placement=True,
                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                        )
config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    print('no previous model param can be used')

    #     files = [file.path for file in os.scandir(save_path) if file.is_file()]

    #     if not files:

    #         sess.run(tf.global_variables_initializer())

    #         print('no previous model param can be used')

    #     else:

    #         saver.restore(sess,tf.train.latest_checkpoint(save_path))

    #         print('use previous model param')

    t_start = time.time()
    for epoch in range(epochs):

        optimizer_GD(x_train, batch_choice=batch_size)

        #         print('epoch = {}'.format(epoch))

        #         train_loss = sess.run(loss,feed_dict={input_x:x_train[0:1]})

        #         print(train_loss)

        data_num = x_train.shape[0]

        # print('data_num = ',data_num)

        # compute training loss
        train_loss = 0

        for j in range(data_num):
            train_loss += sess.run(loss, feed_dict={input_x: x_train[j:j + 1]})

        train_loss /= data_num

        # train_loss = loss_compute(x_train)

        print('train epoch = {}, loss = {}'.format(epoch, train_loss))

        # 紀錄資料:本次epoch ckpt檔
        model_save_path = saver.save(sess, out_dir_prefix, global_step=epoch)

        print('Save model checkpoint to ', model_save_path)

    t_total = time.time() - t_start

    print('training time of {} epochs = {}'.format(epochs, t_total))
