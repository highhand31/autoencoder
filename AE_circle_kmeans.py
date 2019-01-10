import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.contrib.factorization import KMeansClustering
from tensorflow import contrib
# from tensorflow.contrib import slim
# import cv2
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
from tensorflow.contrib.factorization import KMeans
from tensorflow.python.platform import gfile
import pandas as pd
from pandas import ExcelWriter




class AE():

    def __init__(self, input_dim=[None, 28, 28,1], save_path="model_saver\ckpt"):

        self.input_dim = input_dim
        self.save_path = save_path
        self.model_filename = os.path.join(self.save_path,"pb_model.pb")
        self.excel_path = os.path.join(self.save_path,"cf_matrix.xlsx")
        self.kmeans_folder = os.path.join(self.save_path,"kmeans")
        print("save path = ",self.save_path)
        self.kel_x = 9
        self.kel_y = 9
        self.deep = 64
        self.FC1_dropout_ratio = 0.2
        self.FC2_dropout_ratio = 0.4
        self.k_num_iterations = 10
        self.num_cluster = 3
        self.encode_data = []
        self.UDP_init_flag = False
        #self.UDP_init()
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

    def input_fn(self):
        return tf.train.limit_epochs(
            tf.convert_to_tensor(self.encode_data, dtype=tf.float32), num_epochs=1)

    def __build_model(self):
        num_mini = 1e-5
        #self.noise_x = tf.placeholder(tf.float32, self.input_dim, name="noise_x")
        self.input_x = tf.placeholder(tf.float32, self.input_dim, name="input_x")

        self.input_center = tf.placeholder(tf.float32,[None,1024], name="input_center")
        # self.input_cluster_index = tf.placeholder(tf.float32,[None,self.num_cluster], name="input_cluster_index")

        self.prediction = self.__inference(self.input_x)
        encode_shape = self.encode4.shape
        self.input_x_flatten = tf.reshape(self.encode4, (-1, encode_shape[1]*encode_shape[2]*encode_shape[3]))
        self.loss_AE = tf.reduce_mean(tf.pow(self.prediction - self.input_x, 2), name="loss_AE")
        sum_input_x_flatten = tf.reduce_sum(self.input_x_flatten)
        sum_input_center = tf.reduce_sum(self.input_center)
        self.loss_kmeans = tf.reduce_mean(self.input_center/sum_input_center*tf.pow(tf.log(self.input_x_flatten/sum_input_x_flatten+num_mini) -
                                                                   tf.log(self.input_center/sum_input_center+num_mini), 2), name="loss_kmeans")
        # self.loss_kmeans = tf.reduce_sum(self.input_center*(tf.log(self.input_center+num_mini)-tf.log(self.input_x_flatten+num_mini)),name='loss_kmeans')
        # self.loss_kmeans = tf.reduce_mean(tf.pow(self.input_x_flatten - self.input_center, 2), name="loss_kmeans")
        # self.loss_kmeans = tf.distributions.kl_divergence(self.input_center,self.input_x_flatten)

        self.loss = self.loss_AE + self.loss_kmeans
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

        # K-Means Parameters
        num_clusters = 10
        model_dir = r"model_saver\kmeans"
        # self.kmeans = tf.contrib.factorization.KMeansClustering(
        #     num_clusters=num_clusters, use_mini_batch=True, model_dir=None,
        #     initial_clusters=KMeansClustering.KMEANS_PLUS_PLUS_INIT)

        # kmeans = KMeans(inputs=self.k_input, num_clusters=10, distance_metric='squared_euclidean', #'squared_euclidean','cosine'
        #                 use_mini_batch=True)
        # Build KMeans graph
        # training_graph = kmeans.training_graph()
        # (self.all_scores, cluster_idx, self.scores, self.cluster_centers_initialized,
        #  self.cluster_centers_var, self.init_op, self.train_op) = training_graph
        # self.cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
        # self.avg_distance = tf.reduce_mean(self.scores)

        # self.AE_loss = tf.reduce_mean(tf.pow(self.prediction - self.ori_x, 2),name="AE_loss")
        # self.AE_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.AE_optimizer)

        self.save_path = save_path
        self.out_dir_prefix = os.path.join(save_path, "model")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.saver = tf.train.Saver(max_to_keep=20)


    def __inference(self, input_x):
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
                #init:先進行一次inference，收集encode2資料當作是kmeans的輸入
                self.encode_data = []
                for index in range(total_batches):
                    output = sess.run(self.prediction,feed_dict={self.ori_x:origin_data[index:index+1]})
                    self.encode_data.append(self.encode2[0])
                self.encode_data = np.array(self.encode_data)
                for index in range(self.k_num_iterations):
                    self.kmeans.train(self.input_fn)
                    cluster_centers = self.kmeans.cluster_centers()
                    # if previous_centers is not None:
                    #     print('delta:', cluster_centers - previous_centers)
                    # previous_centers = cluster_centers
                    print('iteration {}, score: {}'.format(index,self.kmeans.score(self.input_fn)) )
                # print('cluster centers:', cluster_centers)

                #--Kmeans計算完後的cluster index, center
                cluster_indices = list(self.kmeans.predict_cluster_index(self.input_fn))
                # cluster_index = cluster_indices[i]
                # center = cluster_centers[cluster_index]

                # self.input_center = tf.placeholder(tf.float32,[])
                # self.loss_kmeans =
                #self.AE_loss = tf.reduce_mean(tf.pow(self.prediction - self.ori_x, 2), name="AE_loss")
                # self.AE_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.AE_optimizer)

                for epoch in range(epochs):

                    for index in range(total_batches):

                        num_start = index * batch_size
                        num_end = num_start + batch_size

                        if num_end >= origin_data.shape[0] and batch_size > 1:
                            num_end = origin_data.shape[0] - 1

                        sess.run(self.optimizer, feed_dict={self.ori_x: origin_data[num_start:num_end]
                                                            })

                    # train data mean loss after a epoch
                    train_loss = []
                    for index in range(origin_data.shape[0]):
                        single_loss = sess.run(self.loss, feed_dict={self.ori_x: origin_data[index:index+1]
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

                    graph = tf.get_default_graph().as_graph_def()
                    # output_graph_def = graph_util.convert_variables_to_constants(sess, graph,['output'])  # graph也可以直接填入sess.graph_def
                    output_graph_def = graph_util.convert_variables_to_constants(sess, graph,
                                                                                 ['output/Relu','loss'])  # graph也可以直接填入sess.graph_def


                    # 'model_saver/'為置放的資料夾，'combined_model.pb'為檔名
                    addr_tail = "pb_model.pb"
                    pb_addr = os.path.join(self.save_path,addr_tail)
                    with tf.gfile.GFile(pb_addr, "wb") as f:

                        f.write(output_graph_def.SerializeToString())

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

    def train_AE_kmeans(self, origin_data,label_data, GPU_ratio=0.2, epochs=50, batch_size=1, fine_tune=False,
              save_ckpt=True):

        #origin_data.shpe = (None,28,28,1)
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
                    # self.saver.restore(sess, tf.train.latest_checkpoint(r"model_saver\AE_mnist"))

                    print('use previous model param')

            else:
                sess.run(tf.global_variables_initializer())
                print('no previous model param can be used')


            #init:先進行一次inference，收集encode2資料當作是kmeans的輸入
            # model_filename = r"model_saver\AE_mnist\pb_model.pb"
            if fine_tune is True:
                model_filename = self.model_filename
            else:
                model_filename = r"model_saver\AE_circle\pb_model.pb"
            self.encode_data = self.get_encode_init_data(origin_data,model_filename)#shape = (None,28,28,1)
            shape_encode = self.encode_data.shape

            self.encode_data = np.reshape(self.encode_data,(-1,shape_encode[1]*shape_encode[2]*shape_encode[3]))#進行kmeans的input只能是2維矩陣

            #self.encode_data = np.random.rand(origin_data.shape(0),784)
            print("self.encode_data shape = ",self.encode_data.shape)

            previous_centers = None
            loss_previous = None
            kmeans = tf.contrib.factorization.KMeansClustering(
                num_clusters=self.num_cluster, use_mini_batch=True, model_dir=None,
                initial_clusters=KMeansClustering.RANDOM_INIT)
            for index in range(self.k_num_iterations):
                kmeans.train(self.input_fn)
                cluster_centers = kmeans.cluster_centers()
                loss_kmeans = kmeans.score(self.input_fn)

                print('iteration {}, score: {}'.format(index, loss_kmeans))

                if previous_centers is not None:
                    center_diff_mean = np.mean(cluster_centers - previous_centers)
                    print('center_diff_mean:', center_diff_mean)
                previous_centers = cluster_centers

                if loss_previous is not None:
                    loss_decrease = (loss_kmeans - loss_previous) / loss_previous * 100
                    print("loss decrease % = {}".format(loss_decrease))
                    # check if loss decrease rate under 1%
                    if np.abs(loss_decrease) < 1.0:
                        print("The loss decrease rate is under 1%, stop kmean iteration")
                        break
                loss_previous = loss_kmeans





            #--Kmeans計算完後的cluster index, center
            cluster_indices = list(kmeans.predict_cluster_index(self.input_fn))

            #redistribution of cluster centers
            input_centers = []
            for index in range(origin_data.shape[0]):
                input_centers.append(cluster_centers[cluster_indices[index]])
            input_centers = np.array(input_centers)#shape = (50000,784)

            #input_centers initial by numpy
            # input_centers = np.random.normal(loc=0.0, scale=1.0, size=(origin_data.shape[0],
            #                                                            shape_encode[1]*shape_encode[2]*shape_encode[3]))
            print("input_centers shape = ",input_centers.shape)

            # print(len(cluster_indices))
            # print(len(cluster_centers))

            #origin_data_flatten = np.reshape(origin_data,(-1,784))
            writer = ExcelWriter(self.excel_path)
            for epoch in range(epochs):
                #autoencoder training
                for index in range(total_batches):

                    num_start = index * batch_size
                    num_end = num_start + batch_size

                    if num_end >= origin_data.shape[0] and batch_size > 1:
                        num_end = origin_data.shape[0] - 1

                    #cluster_index = cluster_indices[num_start]
                    sess.run(self.optimizer, feed_dict={self.input_x: origin_data[num_start:num_end],
                                                        self.input_center:input_centers[num_start:num_end]})
                # 紀錄資料:本次epoch ckpt檔
                model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
                print('Save model checkpoint to ', model_save_path)

                #儲存PB檔
                graph = tf.get_default_graph().as_graph_def()
                # output_graph_def = graph_util.convert_variables_to_constants(sess, graph,['output'])  # graph也可以直接填入sess.graph_def
                output_graph_def = graph_util.convert_variables_to_constants(sess, graph,
                                                                             ['output/Relu',
                                                                              "pool4"])  # graph也可以直接填入sess.graph_def
                # addr_tail = "pb_model.pb"
                # pb_addr = os.path.join(self.save_path, addr_tail)
                with tf.gfile.GFile(self.model_filename, "wb") as f:
                    f.write(output_graph_def.SerializeToString())

                '''
                train data mean loss after a epoch
                '''
                train_loss = []
                for index in range(origin_data.shape[0]):
                    # cluster_index = cluster_indices[index]
                    single_loss = sess.run(self.loss, feed_dict={self.input_x: origin_data[index:index + 1],
                                                                 self.input_center: input_centers[index:index + 1]})

                    train_loss.append(single_loss)

                train_loss = np.array(train_loss)
                train_stdv = np.std(train_loss)
                train_loss = np.mean(train_loss)

                # display Epoch information
                msg = "Epoch {}\ntrain set loss = {}".format(epoch, train_loss)
                print(msg)
                '''
                kmeans training
                '''
                # model_filename = r"model_saver\AE_Kmeans\pb_model.pb"


                self.encode_data = self.get_encode_init_data(origin_data, self.model_filename)#shape = (None,28,28,1)
                self.encode_data = np.reshape(self.encode_data, (
                -1, shape_encode[1] * shape_encode[2] * shape_encode[3]))  # 進行kmeans的input只能是2維矩陣
                print("self.encode_data shape = ", self.encode_data.shape)
                previous_centers = None
                loss_previous = None
                kmeans = tf.contrib.factorization.KMeansClustering(
                    num_clusters=self.num_cluster, use_mini_batch=True, model_dir=None,
                    initial_clusters=KMeansClustering.RANDOM_INIT)
                for index in range(self.k_num_iterations):
                    kmeans.train(self.input_fn)
                    cluster_centers = kmeans.cluster_centers()
                    loss_kmeans = kmeans.score(self.input_fn)

                    print('iteration {}, score: {}'.format(index, loss_kmeans))

                    if previous_centers is not None:
                        center_diff_mean = np.mean(cluster_centers - previous_centers)
                        print('center_diff_mean:', center_diff_mean)
                    previous_centers = cluster_centers

                    if loss_previous is not None:
                        loss_decrease = (loss_kmeans - loss_previous) / loss_previous * 100
                        print("loss decrease % = {}".format(loss_decrease))
                        # check if loss decrease rate under 1%
                        if np.abs(loss_decrease) < 1.0:
                            print("The loss decrease rate is under 1%, stop kmean iteration")
                            break
                    loss_previous = loss_kmeans



                # --Kmeans計算完後的cluster index, center
                cluster_indices = list(kmeans.predict_cluster_index(self.input_fn))

                '''
                confusion matrix save
                '''
                k_indice = np.array(cluster_indices)
                plot = pd.crosstab(k_indice, label_data, rownames=['k'], colnames=['label'])
                plot.to_excel(writer, 'Epoch' + str(epoch))

                # redistribution of cluster centers
                input_centers = []
                for index in range(origin_data.shape[0]):
                    input_centers.append(cluster_centers[cluster_indices[index]])
                input_centers = np.array(input_centers)  # shape = (50000,784)
                print("input_centers shape = ", input_centers.shape)



            writer.save()

    def get_encode_init_data(self,input_data,model_filename):
        # model_filename = r"model_saver\AE_mnist\pb_model.pb"
        # 設定GPU參數
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                )
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        with tf.Session() as sess:
            with gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()

                tf.import_graph_def(graph_def, name='')  # 導入計算圖

            sess.run(tf.global_variables_initializer())

            input_x = sess.graph.get_tensor_by_name("input_x:0")
            # supressed_x = sess.graph.get_tensor_by_name("supressed_x:0")
            # print(input_x.shape)
            # result = sess.graph.get_tensor_by_name("output/Relu:0")
            encode2 = sess.graph.get_tensor_by_name("pool4:0")
            # print(result.shape)

            encode_data = []
            for index in range(input_data.shape[0]):
                # reconstruct = sess.run(result, feed_dict={input_x: input_data[test_num:test_num + 1]})
                temp = sess.run(encode2, feed_dict={input_x: input_data[index:index + 1]})
                encode_data.append(temp[0])
            encode_data = np.array(encode_data)
            print("encode_data shape = ",encode_data.shape)
        return encode_data





if __name__ == "__main__":
    save_path = r"model_saver\AE_pill_KL_Kmeans"
    out_dir_prefix = os.path.join(save_path, "model")
    height = 64
    width = 64
    epochs = 30
    GPU_ratio = 0.4
    batch_size = 1
    train_ratio = 1.0

    #DAE train

    # pic_path = r'E:\dataset\forAE\circle\train\Good'
    # (x_ori, x_ori_label, no_care, no_care_2) = cm.data_load(pic_path, train_ratio=train_ratio, resize=(width, height),
    #                                                             normalize=False,has_dir=False)
    # print(x_ori.shape)
    #
    # x_noise = []
    # x_ori_2 = []
    #
    # #augmentation
    # for index in range(x_ori.shape[0]):
    #     x_ori_2.append(x_ori[index])
    #     x_ori_2.append(x_ori[index])
    # x_ori_2 = np.array(x_ori_2)
    # print(x_ori_2.shape)
    # #讓一半的資料有增噪，另一半是沒有的
    # for index in range(x_ori_2.shape[0]):
    #     if index % 2:
    #         img = skimage.util.random_noise(x_ori_2[index], mode='gaussian', seed=None, clip=True)
    #     else:
    #         img = x_ori_2[index]
    #     x_noise.append(img)
    #     #print(index)
    #
    # x_noise = np.array(x_noise)
    # x_noise = np.float32(x_noise)#執行完增噪後為float64，轉成float32，否則丟到input裡會出問題
    # # x_noise = int(x_noise)
    # print("x_ori_2 shape = ", x_ori_2.shape)
    # print('x_ori_2 dtype = ', x_ori_2.dtype)
    # # print('x_ori content = ', x_ori[0])
    # print("x_noise shape = ",x_noise.shape)
    # print('x_noise dtype = ',x_noise.dtype)
    # print('x_noise content = ',x_noise[0])

    #確定2個圖片的差異
    # plot = plt.figure()
    # plt.subplot(2,1,1)
    # plt.imshow(x_ori[1])
    #
    # plt.subplot(2, 1, 2)
    # plt.imshow(x_noise[1])
    #
    # plt.show()

    # prepare test data
    # pic_path = r'E:\dataset\forAE\pill\test'
    # pic_path = r'E:\dataset\forAE\circle\test'
    # (no_care,no_care_2,x_test,x_test_label) = cm.data_load(pic_path,train_ratio=0,resize=(width,height),
    #                                                        shuffle = True,normalize = False)
    # # # print('x_train shape = ',x_train_2.shape)
    # # # print('x_train_label shape = ',x_train_label_2.shape)
    # print('x_test shape = ',x_test.shape)
    # print('x_test_label shape = ',x_test_label.shape)
    # print(np.max(x_test_label))
    #
    # ae = AE(input_dim=[None, width, height, 3], save_path="model_saver\ckpt")
    # ae.train(x_ori_2,x_noise,x_test,x_test_label,GPU_ratio=GPU_ratio,epochs = epochs,fine_tune=False)
    # print("AI訓練結束")


    #mnist data preprocess
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    # input_mnist = mnist.train.images
    # label_mnist = mnist.train.labels
    # input_mnist = np.reshape(input_mnist,(-1,28,28,1))
    # print(input_mnist.shape)

    #circle data preprocess
    # -->train data
    # pic_path = r'E:\dataset\Surface_detection'
    pic_path = r'E:\dataset\Halcon_Pill\Original'
    (input_train, input_train_label, no_care, no_care_2) = cm.data_load(pic_path, train_ratio=1, resize=(64, 64),
                                                             shuffle=True, normalize=True, has_dir=True)

    print('x_train shape = ', input_train.shape)
    print('x_train_label shape = ', input_train_label.shape)
    # print(input_train_label)

    ae = AE(input_dim=[None, width, height,3], save_path=save_path)
    #temp = ae.get_encode_init_data(input_mnist)
    #print(temp.shape)
    ae.train_AE_kmeans(input_train,input_train_label, GPU_ratio=GPU_ratio, epochs=epochs, fine_tune=False)
    print("AI訓練結束")