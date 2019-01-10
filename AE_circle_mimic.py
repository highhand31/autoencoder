import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow import contrib
# from tensorflow.contrib import slim
import cv2
import os
import numpy as np
import socket as sc
# import time
# import matplotlib.pyplot as plt
import common as cm
import json
import csv
import skimage
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
import test_fun




class AE():

    def __init__(self, input_dim=[None, 64, 64, 3], save_path="model_saver\ckpt"):

        self.input_dim = input_dim
        self.save_path = save_path
        self.kel_x = 9
        self.kel_y = 9
        self.deep = 64
        self.deep_mimic = int(self.deep/2)
        self.UDP_init_flag = False
        self.UDP_init()
        #self.data_init()
        self.__build_model()
        #self.train(self.x_train,self.x_test_2,self.x_test_label_2, epochs=10, fine_tune=False)

        #print("initial")

    def data_init(self):
        save_path = r"model_saver\AE_circle"
        out_dir_prefix = os.path.join(save_path, "model")
        height = 64
        width = 64
        epochs = 50
        GPU_ratio = 0.5
        batch_size = 12
        train_ratio = 0.7

        pic_path = r'./goodsamples'
        (self.x_train, self.x_train_label, no1, no2) = cm.data_load(pic_path, train_ratio=1,
                                                                    resize=(width, height),
                                                                    has_dir=False)
        print(self.x_train.shape)
        # self.ui.plainTextEdit.appendPlainText('x_train.shape = {}'.format(x_train.shape))
        print('x_train.shape = {}'.format(self.x_train.shape))
        # self.ui.label.setText('x_train.shape = {}'.format(x_train.shape))

        # prepare test data
        pic_path = r'./xxx'
        (self.x_train_2, self.x_train_label_2, self.x_test_2, self.x_test_label_2) = cm.data_load(pic_path, train_ratio,
                                                                              resize=(width, height), shuffle=True,
                                                                              normalize=True)

        print('x_test shape = ', self.x_test_2.shape)
        # self.ui.plainTextEdit.appendPlainText('x_test shape = {}'.format(x_test_2.shape))
        print('x_test shape = {}'.format(self.x_test_2.shape))

        print('x_test_label shape = ', self.x_test_label_2.shape)
        # self.ui.plainTextEdit.appendPlainText('xx_test_label shape = {}'.format(x_test_label_2.shape))
        print('x_test_label shape = {}'.format(self.x_test_label_2.shape))


    def UDP_init(self):
        self.sock = sc.socket(sc.AF_INET, sc.SOCK_DGRAM)
        self.send_address = ("127.0.0.1",5002)
        #json format {"mode":str,"msg":str, "train loss":list, "acc":list}
        #if UDP transmit msg -->"mode":"msg"
        #if UDP transmit training data -->"mode":"dict"
        send_msg = {"mode":"msg","msg":"AI Model UDP TX init ok"}
        send_msg = json.dumps(send_msg)
        self.UDP_send(send_msg,self.send_address)
        self.UDP_dict = {}
        self.UDP_init_flag = True

    def UDP_send(self,send_msg,send_address):

        send_msg = bytes(send_msg, encoding="gbk")
        self.sock.sendto(send_msg, send_address)

    def __build_model(self):

        # self.noise_x = tf.placeholder(tf.float32, self.input_dim, name="noise_x")
        self.input_x = tf.placeholder(tf.float32, self.input_dim, name="input_x")

        self.reconstruct = self.__inference(self.input_x)
        self.squeeze = self.__mimic(self.reconstruct)
        self.loss1 = tf.reduce_mean(tf.pow(self.reconstruct - self.input_x, 2),name="loss1")#input_x為有加noise的圖片
        self.loss_mimic = tf.reduce_mean(tf.pow(self.squeeze - self.reconstruct, 2), name="loss_mimic")
        self.loss = self.loss1 + self.loss_mimic
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

        self.save_path = save_path
        self.out_dir_prefix = os.path.join(save_path, "model")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.saver = tf.train.Saver(max_to_keep=20)


    def __inference(self, input_x):

        print("----reconstruction inference----")

        net = tf.layers.conv2d(
            inputs=input_x,
            filters=self.deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net, pool1_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        self.encode1 = net

        print("encode_1 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # net shape = 32 x 32 x 64

        net = tf.layers.conv2d(
            inputs=net,
            filters=self.deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net, pool2_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        self.encode2 = net

        print("encode_2 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # data= 16 x 16 x 64

        net = tf.layers.conv2d(
            inputs=net,
            filters=self.deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net, pool3_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        self.encode3 = net

        print("encode_3 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # data= 8 x 8 x 64

        net = tf.layers.conv2d(
            inputs=net,
            filters=self.deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net, pool4_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        self.encode4 = net

        print("encode_4 shape = ", net.shape)
        # -----------------------------------------------------------------------

        # data= 4 x 4 x 64

        # net = tf.layers.conv2d_transpose(net,64,[2,2],strides=2,padding='VALID',
        #                                 #bias_initializer=tf.ones_initializer(),
        #                                 #kernel_initializer=tf.ones_initializer()
        #                                 )
        net = self.unpool_with_argmax(net, pool4_indices, batch_size, name="Unpool1", ksize=[1, 2, 2, 1])

        net = tf.layers.conv2d(
            inputs=net,
            filters=self.deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        decode1 = net

        print("decode_1 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # data= 8 x 8 x 64

        # net = tf.layers.conv2d_transpose(net,64,[2,2],strides=2,padding='VALID',
        #                                 #bias_initializer=tf.ones_initializer(),
        #                                 #kernel_initializer=tf.ones_initializer()
        #                                 )
        net = self.unpool_with_argmax(net, pool3_indices, batch_size, name="Unpool2", ksize=[1, 2, 2, 1])

        net = tf.layers.conv2d(
            inputs=net,
            filters=self.deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        decode2 = net

        print("decode_2 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # data= 16 x 16 x 64

        # net = tf.layers.conv2d_transpose(net,64,[2,2],strides=2,padding='VALID',
        #                                 #bias_initializer=tf.ones_initializer(),
        #                                 #kernel_initializer=tf.ones_initializer()
        #                                 )
        net = self.unpool_with_argmax(net, pool2_indices, batch_size, name="Unpool3", ksize=[1, 2, 2, 1])

        net = tf.layers.conv2d(
            inputs=net,
            filters=self.deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        self.decode3 = net

        print("decode_3 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # data= 32 x 32 x 64

        # net = tf.layers.conv2d_transpose(net,64,[2,2],strides=2,padding='VALID',
        #                                 #bias_initializer=tf.ones_initializer(),
        #                                 #kernel_initializer=tf.ones_initializer()
        #                                 )

        net = self.unpool_with_argmax(net, pool1_indices, batch_size, name="Unpool4", ksize=[1, 2, 2, 1])

        net = tf.layers.conv2d(
            inputs=net,
            filters=3,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu,
            name="output")

        self.decode4 = net

        print("decode_4 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # data= 64 x 64 x 3

        return net

    def __mimic(self,input_x):
        kernel_deep = self.deep_mimic

        print("----mimic structure----")
        net1 = tf.layers.conv2d(
            inputs=input_x,
            filters=kernel_deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net1, pool1_indices = tf.nn.max_pool_with_argmax(net1, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='mimic_pool1')

        #self.encode1 = net1

        print("encode_1 shape = ", net1.shape)
        # -----------------------------------------------------------------------
        # net shape = 32 x 32 x 64

        net1 = tf.layers.conv2d(
            inputs=net1,
            filters=kernel_deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net1, pool2_indices = tf.nn.max_pool_with_argmax(net1, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='mimic_pool2')

        #self.encode2 = net1

        print("encode_2 shape = ", net1.shape)
        # -----------------------------------------------------------------------
        # data= 16 x 16 x 64

        net1 = tf.layers.conv2d(
            inputs=net1,
            filters=kernel_deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net1, pool3_indices = tf.nn.max_pool_with_argmax(net1, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='mimic_pool3')

        #self.encode3 = net1

        print("encode_3 shape = ", net1.shape)
        # -----------------------------------------------------------------------
        # data= 8 x 8 x 64

        net1 = tf.layers.conv2d(
            inputs=net1,
            filters=kernel_deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net1, pool4_indices = tf.nn.max_pool_with_argmax(net1, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='mimic_pool4')

       # self.encode4 = net1

        print("encode_4 shape = ", net1.shape)
        # -----------------------------------------------------------------------

        # data= 4 x 4 x 64

        # net = tf.layers.conv2d_transpose(net,64,[2,2],strides=2,padding='VALID',
        #                                 #bias_initializer=tf.ones_initializer(),
        #                                 #kernel_initializer=tf.ones_initializer()
        #                                 )
        net1 = self.unpool_with_argmax(net1, pool4_indices, batch_size, name="mimic_Unpool1", ksize=[1, 2, 2, 1])

        net1 = tf.layers.conv2d(
            inputs=net1,
            filters=kernel_deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        decode1 = net1

        print("decode_1 shape = ", net1.shape)
        # -----------------------------------------------------------------------
        # data= 8 x 8 x 64

        # net = tf.layers.conv2d_transpose(net,64,[2,2],strides=2,padding='VALID',
        #                                 #bias_initializer=tf.ones_initializer(),
        #                                 #kernel_initializer=tf.ones_initializer()
        #                                 )
        net1 = self.unpool_with_argmax(net1, pool3_indices, batch_size, name="mimic_Unpool2", ksize=[1, 2, 2, 1])

        net1 = tf.layers.conv2d(
            inputs=net1,
            filters=kernel_deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        #decode2 = net1

        print("decode_2 shape = ", net1.shape)
        # -----------------------------------------------------------------------
        # data= 16 x 16 x 64

        # net = tf.layers.conv2d_transpose(net,64,[2,2],strides=2,padding='VALID',
        #                                 #bias_initializer=tf.ones_initializer(),
        #                                 #kernel_initializer=tf.ones_initializer()
        #                                 )
        net1 = self.unpool_with_argmax(net1, pool2_indices, batch_size, name="mimic_Unpool3", ksize=[1, 2, 2, 1])

        net1 = tf.layers.conv2d(
            inputs=net1,
            filters=kernel_deep,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        #self.decode3 = net1

        print("decode_3 shape = ", net1.shape)
        # -----------------------------------------------------------------------
        # data= 32 x 32 x 64

        # net = tf.layers.conv2d_transpose(net,64,[2,2],strides=2,padding='VALID',
        #                                 #bias_initializer=tf.ones_initializer(),
        #                                 #kernel_initializer=tf.ones_initializer()
        #                                 )

        net1 = self.unpool_with_argmax(net1, pool1_indices, batch_size, name="mimic_Unpool4", ksize=[1, 2, 2, 1])

        net1 = tf.layers.conv2d(
            inputs=net1,
            filters=3,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu,
            name="mimic_output")

        #self.decode4 = net1

        print("decode_4 shape = ", net1.shape)
        # -----------------------------------------------------------------------
        # data= 64 x 64 x 3

        return net1

    def unpool_with_argmax(self,pool, ind, batch_size, name=None, ksize=[1, 2, 2, 1]):
        """
           Unpooling layer after max_pool_with_argmax.
           Args:
               pool:   max pooled output tensor
               ind:      argmax indices
               ksize:     ksize is the same as for the pool
           Return:
               unpool:    unpooling tensor
        """
        with tf.variable_scope(name):

            input_shape = pool.get_shape().as_list()
            output_shape = (batch_size, input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

            flat_output_shape = [batch_size, output_shape[1] * output_shape[2] * output_shape[3]]
            flat_input_size = np.prod(batch_size * input_shape[1] * input_shape[2] * input_shape[3])

            pool_ = tf.reshape(pool, [flat_input_size])
            batch_range = tf.reshape(tf.range(batch_size, dtype=ind.dtype), shape=[batch_size, 1, 1, 1])
            b = tf.ones_like(ind) * batch_range
            b = tf.reshape(b, [flat_input_size, 1])
            ind_ = tf.reshape(ind, [flat_input_size, 1])
            ind_ = tf.concat([b, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
            ret = tf.reshape(ret, output_shape)
            return ret

    def train(self, origin_data, GPU_ratio=0.2, epochs=50, batch_size=1, fine_tune=False,
              save_ckpt=True):

        #             self.GPU_ratio = GPU_ratio
        #             self.epochs = epochs
        #             self.batch_size = batch_size

        #var init
        self.train_loss_set = []
        self.acc_set = []

        # 計算total batch
        total_batches = origin_data.shape[0] // batch_size
        if origin_data.shape[0] % batch_size:
            total_batches += 1

        # 設定GPU參數
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                )
        config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

        with tf.Session(config=config) as sess:

            print("開始與GPU做連結，這部分會花比較久時間，請耐心等候")

            if fine_tune is True:  # 使用已經訓練好的權重繼續訓練
                files = [file.path for file in os.scandir(self.save_path) if file.is_file()]

                if not files:  # 沒有任何之前的權重

                    sess.run(tf.global_variables_initializer())

                    print('no previous model param can be used')

                else:

                    self.saver.restore(sess, tf.train.latest_checkpoint(self.save_path))

                    print('use previous model param')

            else:
                sess.run(tf.global_variables_initializer())
                print('no previous model param can be used')



            try:
                for epoch in range(epochs):

                    for index in range(total_batches):

                        num_start = index * batch_size
                        num_end = num_start + batch_size

                        if num_end >= origin_data.shape[0] and batch_size > 1:
                            num_end = origin_data.shape[0] - 1

                        sess.run(self.optimizer, feed_dict={self.input_x: origin_data[num_start:num_end]
                                                            })

                    # train data mean loss after a epoch
                    train_loss = []
                    for index in range(origin_data.shape[0]):
                        single_loss = sess.run(self.loss, feed_dict={self.input_x: origin_data[index:index+1]
                                                                     })

                        train_loss.append(single_loss)

                    train_loss = np.array(train_loss)
                    train_stdv = np.std(train_loss)
                    train_loss = np.mean(train_loss)

                    #record train loss in the csv file
                    # file_name = "train_notes.csv"
                    # with open(file_name,"w") as csvFile:
                    #     fields = ["average loss","stdv"]
                    #     dictWriter = csv.DictWriter(csvFile,fieldnames=fields)
                    #     dictWriter.writeheader()
                    #     dictWriter.writerow({"average loss":train_loss, "stdv":train_stdv})


                    # test data mean loss after a epoch
                    # test_loss = []
                    # acc = 0
                    # for index in range(test_data.shape[0]):
                    #
                    #     single_loss = sess.run(self.loss, feed_dict={self.ori_x: test_data[index:index+1],
                    #                                                  self.noise_x: test_data[index:index+1]})
                    #     #print('test single_loss = ',single_loss)
                    #     test_loss.append(single_loss)
                    #     if single_loss > train_loss:
                    #
                    #         if test_label[index] > 0:
                    #             acc += 1
                    #
                    #     elif single_loss >= 0:
                    #
                    #         if test_label[index] == 0:
                    #             acc += 1
                    #
                    # acc /= test_data.shape[0]
                    # test_loss = np.array(test_loss)
                    # test_loss = np.mean(test_loss)

                    #display Epoch information
                    msg = "Epoch {}\ntrain set loss = {}".format(epoch, train_loss)
                    print(msg)
                    #self.UDP_send(msg,self.send_address)

                    # msg = "test set loss = {}, accuracy = {}".format(test_loss, acc)
                    # print(msg)
                    #self.UDP_send(msg, self.send_address)

                    # 紀錄資料:本次epoch ckpt檔
                    if save_ckpt is True:
                        model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)

                        print('Save model checkpoint to ', model_save_path)

                    # 紀錄資料:pb檔
                    graph = tf.get_default_graph().as_graph_def()
                    # output_graph_def = graph_util.convert_variables_to_constants(sess, graph,['output'])  # graph也可以直接填入sess.graph_def
                    output_graph_def = graph_util.convert_variables_to_constants(sess, graph,
                                                                                 ['output/Relu',"pool4","mimic_output/Relu"])  # graph也可以直接填入sess.graph_def

                    # 'model_saver/'為置放的資料夾，'combined_model.pb'為檔名
                    addr_tail = "pb_model.pb"
                    pb_addr = os.path.join(self.save_path, addr_tail)
                    with tf.gfile.GFile(pb_addr, "wb") as f:

                        f.write(output_graph_def.SerializeToString())

                    #讀取PB檔製造reconstruct及mimic照片
                    # self.read_pb(pb_addr,epoch)
                    test_fun.read_pb(pb_addr,epoch)


                    #training data collection
                    # 原本train_loss的型態是numpy.float32，進行json序列化會失敗，要改成python內建的float型態才能進行json序列化


                    # print("type of train loss = {}".format(type(train_loss)))
                    # print("type of acc = {}".format(type(acc)))
                    #self.acc_set.append(acc)

                    # self.UDP_dict["train loss"] = self.train_loss_set
                    # self.UDP_dict["acc"] = self.acc_set
                    # for name,value in self.UDP_dict.items():
                    #     print("{} = {}".format(name,value))

                    #UDP sender
                    if self.UDP_init_flag:
                        # self.train_loss_set.append(float(train_loss))
                        # self.acc_set.append(acc)
                        self.UDP_dict["mode"] = "dict"
                        self.UDP_dict["train loss"] = self.train_loss_set
                        self.UDP_dict["acc"] = self.acc_set
                        jsonObj = json.dumps(self.UDP_dict)#若要傳送dict，需要先進行json化
                        # jsonObj = json.dumps(self.train_loss_set)
                        # self.UDP_send("UDP sender:loss = {},acc = {}".format(train_loss,acc),self.send_address)
                        self.UDP_send(jsonObj,self.send_address)

                    # return train_loss, acc
            except:
                print("執行訓練時出現錯誤，可能在使用fine tune時，height,width與模型不符合")

    def read_pb(self,pb_addr,epoch):
        model_filename = r"model_saver\AE_circle_mimic\pb_model.pb"
        # model_filename = pb_addr

        # pic_path = r'E:\dataset\Surface_detection'
        pic_path = r'E:\dataset\Surface_detection\crack\pill_magnesium_crack_069.png'

        x_train=[]
        img = cv2.imread(pic_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(64,64))

        x_train.append(img)
        x_train = np.array(x_train)
        x_train = x_train/255
        print("L1     OK")

        print('x_train shape = ', x_train.shape)
        # print('x_train_label shape = ', input_train_label.shape)
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
                print("L2     OK")

                tf.import_graph_def(graph_def, name='')  # 導入計算圖

            sess.run(tf.global_variables_initializer())

            ori_x = sess.graph.get_tensor_by_name("input_x:0")
            # supressed_x = sess.graph.get_tensor_by_name("supressed_x:0")
            # print(input_x.shape)
            output_re = sess.graph.get_tensor_by_name("output/Relu:0")
            output_mimic = sess.graph.get_tensor_by_name("mimic_output/Relu:0")
            print("L3     OK")
            # print(result.shape)

            # test_num = 22
            reconstruct = sess.run(output_re, feed_dict={ori_x: x_train[0:1]})
            mimic = sess.run(output_mimic, feed_dict={ori_x: x_train[0:1]})
            print("reconstruct.shape = ", reconstruct.shape)
            print("encode_data.shape = ", mimic.shape)
            print("L4     OK")

            #save picture
            reconstruct = reconstruct[0]*255
            mimic = mimic[0]*255
            print("L5     OK")
            reconstruct = cv2.cvtColor(reconstruct,cv2.COLOR_RGB2BGR)
            mimic = cv2.cvtColor(mimic, cv2.COLOR_RGB2BGR)
            print("L6     OK")
            # cv2.imwrite(os.path.join(self.save_path,"epoch"+str(epoch)+"_"+"recon.png"),reconstruct)
            # cv2.imwrite(os.path.join(self.save_path, "epoch" + str(epoch) + "_" + "mimic.png"), mimic)
            cv2.imwrite("epoch" + str(epoch) + "_" + "recon.png", reconstruct)
            cv2.imwrite("epoch" + str(epoch) + "_" + "mimic.png", mimic)


    def eval(self, train_data, test_data, test_label, GPU_ratio=0.2):

        total_loss = []
        single_loss = 0

        # 設定GPU參數
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                )
        config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

        with tf.Session(config=config) as sess:

            self.saver.restore(sess, tf.train.latest_checkpoint(self.save_path))

            print('use previous model param')

            for i in range(train_data.shape[0]):
                single_loss = sess.run(self.loss, feed_dict={self.input_x: train_data[i:i + 1]})

                total_loss.append(single_loss)

            total_loss = np.array(total_loss)

            train_ave_loss = np.mean(total_loss)

            total_loss = []

            acc = 0

            for i in range(test_data.shape[0]):

                single_loss = sess.run(self.loss, feed_dict={self.input_x: test_data[i:i + 1]})

                if single_loss > train_ave_loss:

                    if test_label[i] == 0 or test_label[i] == 1:
                        acc += 1

                elif single_loss >= 0:

                    if test_label[i] == 2:
                        acc += 1

                total_loss.append(single_loss)

            acc /= test_data.shape[0]

            total_loss = np.array(total_loss)

            test_ave_loss = np.mean(total_loss)

            return train_ave_loss, test_ave_loss, acc

if __name__ == "__main__":
    save_path = r"model_saver\AE_circle_mimic"
    height = 64
    width = 64
    epochs = 20
    GPU_ratio = 0.5
    batch_size = 1
    train_ratio = 1.0

    #AE train
    pic_path = r'E:\dataset\Surface_detection'
    (input_train, input_train_label, no_care, no_care_2) = cm.data_load(pic_path, train_ratio=1, resize=(64, 64),
                                                                        shuffle=True, normalize=True, has_dir=True)

    print('x_train shape = ', input_train.shape)
    print('x_train_label shape = ', input_train_label.shape)
    print(input_train_label)

    ae = AE(input_dim=[None, width, height, 3], save_path=save_path)
    ae.train(input_train, GPU_ratio=GPU_ratio, epochs=epochs, fine_tune=False)
    print("AI訓練結束")

