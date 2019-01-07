import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow import contrib
# from tensorflow.contrib import slim
import cv2
import os
import numpy as np
import socket as sc
import time
# import matplotlib.pyplot as plt
import common as cm
import json
import csv
import skimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




class AE():

    def __init__(self, input_dim=[None, 300, 300, 3], save_path="model_saver\ckpt"):

        self.input_dim = input_dim
        self.save_path = save_path
        self.kel_x = 5
        self.kel_y = 5
        self.deep = 64
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

        self.supressed_x = tf.placeholder(tf.float32, self.input_dim, name="supressed_x")
        self.ori_x = tf.placeholder(tf.float32, [None,256,256,3], name="ori_x")

        self.prediction = self.__inference(self.supressed_x)
        self.loss = tf.reduce_mean(tf.pow(self.prediction - self.ori_x, 2),name="loss")#input_x為有加noise的圖片
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

        self.save_path = save_path
        self.out_dir_prefix = os.path.join(save_path, "model")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.saver = tf.train.Saver(max_to_keep=20)


    def __inference(self, input_x):

        # input_data = tf.image.resize_images(input_x, [128, 128], method=0)
        #input_data = tf.image.resize_images(input_data,[256,256],method=0)

        net = tf.layers.conv2d(
            inputs=input_x,
            filters=64,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net, pool1_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        encode1 = net

        print("encode_1 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # net shape = 32 x 32 x 64

        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net, pool2_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        encode2 = net

        print("encode_2 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # data= 16 x 16 x 64

        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net, pool3_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        encode3 = net

        print("encode_3 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # data= 8 x 8 x 64

        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        # net=tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net, pool4_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        encode4 = net

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
            filters=64,
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
            filters=64,
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
            filters=64,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        decode3 = net

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
            filters=64,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu
            )

        decode4 = net

        print("decode_4 shape = ", net.shape)
        # -----------------------------------------------------------------------
        # data= 64 x 64 x 3

        net = tf.layers.conv2d_transpose(net,64,[2,2],strides=2,padding='VALID',
                                        #bias_initializer=tf.ones_initializer(),
                                        #kernel_initializer=tf.ones_initializer()
                                        )

        net = tf.layers.conv2d(
            inputs=net,
            filters=3,
            kernel_size=[self.kel_x, self.kel_y],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu,
            name="output")

        decode5 = net

        print("decode_5 shape = ", net.shape)

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

    def train(self, origin_data_addr, GPU_ratio=0.2, epochs=50, batch_size=1, fine_tune=False,
              save_ckpt=True):

        #             self.GPU_ratio = GPU_ratio
        #             self.epochs = epochs
        #             self.batch_size = batch_size

        #var init
        self.train_loss_set = []
        self.acc_set = []

        # 計算total batch
        total_batches = origin_data_addr.shape[0] // batch_size
        if origin_data_addr.shape[0] % batch_size:
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
                    start_time = time.time()

                    for index in range(total_batches):

                        num_start = index * batch_size
                        num_end = num_start + batch_size

                        if num_end >= origin_data_addr.shape[0] and batch_size > 1:
                            num_end = origin_data_addr.shape[0] - 1

                        #read picture
                        input_ori = []
                        input_suppressed = []
                        for index_2 in range(num_start,num_end):
                            # img = mpimg.imread(origin_data[index_2])
                            img = cv2.imread(origin_data_addr[index_2])
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            if img.shape[1] != 256 or img.shape[2] != 256:
                                img = cv2.resize(img,(256,256))
                            input_ori.append(img)

                            img = cv2.resize(img,(self.input_dim[2],self.input_dim[1]))

                            input_suppressed.append(img)

                        input_ori = np.array(input_ori)
                        input_ori = input_ori.astype("float32")
                        input_ori = input_ori / 255

                        input_suppressed = np.array(input_suppressed)
                        input_suppressed = input_suppressed.astype("float32")
                        input_suppressed = input_suppressed / 255



                        sess.run(self.optimizer, feed_dict={self.ori_x:input_ori,self.supressed_x:input_suppressed})

                    # train data mean loss after a epoch
                    train_loss = []

                    for index in range(origin_data_addr.shape[0]):
                        input_ori = []
                        input_suppressed = []
                        img = cv2.imread(origin_data_addr[index])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # img = mpimg.imread(origin_data_addr[index])
                        input_ori.append(img)

                        img = cv2.resize(img, (self.input_dim[2], self.input_dim[1]))
                        input_suppressed.append(img)

                        input_ori = np.array(input_ori)
                        input_ori = input_ori.astype("float32")
                        input_ori = input_ori / 255

                        input_suppressed = np.array(input_suppressed)
                        input_suppressed = input_suppressed.astype("float32")
                        input_suppressed = input_suppressed / 255

                        single_loss = sess.run(self.loss, feed_dict={self.ori_x:input_ori,self.supressed_x:input_suppressed})

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
                    #                                                  })
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

                    msg = "Epoch {} takes time = {}".format(epoch,time.time()-start_time)
                    print(msg)

                    # 紀錄資料:本次epoch ckpt檔
                    if save_ckpt is True:
                        model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)

                        print('Save model checkpoint to ', model_save_path)

                    graph = tf.get_default_graph().as_graph_def()
                    # output_graph_def = graph_util.convert_variables_to_constants(sess, graph,['output'])  # graph也可以直接填入sess.graph_def
                    output_graph_def = graph_util.convert_variables_to_constants(sess, graph,
                                                                                 ['output/Relu','loss'])  # graph也可以直接填入sess.graph_def


                    # 'model_saver/'為置放的資料夾，'combined_model.pb'為檔名
                    with tf.gfile.GFile("model_saver/pb_test_model.pb", "wb") as f:

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

if __name__ == "__main__":
    save_path = r"model_saver\SuperR"
    out_dir_prefix = os.path.join(save_path, "model")
    height = 128
    width = 128
    epochs = 50
    GPU_ratio = 0.8
    batch_size = 1
    train_ratio = 1.0

    #super resolution train

    pic_path = r'E:\dataset\For_super_resolution\train'
    # pic_path = r'E:\dataset\XXX\SR'
    (x_ori, x_ori_label, no_care, no_care_2) = cm.data_load(pic_path, train_ratio=train_ratio,
                                                                normalize=False,has_dir=False,Only_addr=True)
    # print(type(x_ori),len(x_ori))

    #因為圖片有10萬張，只能丟位址進去，取batch時再讀取圖片
    print("x_ori shape = ", x_ori.shape)
    print('x_ori dtype = ', type(x_ori))
    print('x_ori content 0 = ', x_ori[:10])


    # print('x_ori content = ', x_ori[0])


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
    # # print('x_train shape = ',x_train_2.shape)
    # # print('x_train_label shape = ',x_train_label_2.shape)
    # print('x_test shape = ',x_test.shape)
    # print('x_test_label shape = ',x_test_label.shape)

    ae = AE(input_dim=[None, width, height, 3], save_path=save_path)
    ae.train(x_ori,GPU_ratio=GPU_ratio,epochs = epochs,fine_tune=True)
    print("AI訓練結束")

