import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import common as cm

height = 300
width = 428
epochs = 50
GPU_ratio = 0.2
batch_size = 12
train_ratio = 0.7

pic_path_train = r'E:\dataset\Halcon_Pill\Original\good'
pic_path_test = r'E:\dataset\Halcon_Pill\Original'
save_path = r"model_saver\AE_pill"



#將所有的Good sample當作是training set
(x_train,x_train_label,np1,np2) = cm.data_load(pic_path_train,train_ratio=1,resize=(width,height),has_dir=False)
print('x_train shape = ',x_train.shape)
print('x_train_label shape = ',x_train_label.shape)
# print('x_test shape = ',np1.shape)
# print('x_test_label shape = ',np2.shape)

#將good,crack,contaminaion依照比例取成test set
(x_test,x_test_label,x_test_2,x_test_2_label) = cm.data_load(pic_path_test,train_ratio,resize=(width,height))
print('x_test shape = ',x_test.shape)
print('x_test_label shape = ',x_test_label.shape)
print('x_test_2 shape = ',x_test_2.shape)
print('x_test_2_label shape = ',x_test_2_label.shape)

saver = tf.train.Saver(max_to_keep=20)

# 設定GPU參數
config = tf.ConfigProto(log_device_placement=True,
                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                        )
config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

with tf.Session(config=config) as sess:
    files = [file.path for file in os.scandir(save_path) if file.is_file()]

    if not files:

        sess.run(tf.global_variables_initializer())

        print('no previous model param can be used')

    else:

        saver.restore(sess, tf.train.latest_checkpoint(save_path))

        print('use previous model param')

    # compute training loss
    train_loss = []

    for j in range(x_train.shape[0]):
        a = sess.run(loss, feed_dict={input_x: x_train[j:j + 1]})

        train_loss.append(a)

    train_loss = np.array(train_loss)

    train_std = np.std(train_loss)

    train_mean = np.mean(train_loss)

    print('train loss std = ', train_std)
    print('train loss mean = ', train_mean)

    # pic(all 1) test
    #     test_pic = np.ones([1,flatten_num],dtype="float32")

    #     net2 = sess.run(net,feed_dict={input_x:test_pic})

    #     net2 = net2*255

    #     net2 = net2.astype("int")

    #     plt.figure(figsize=(12, 12))

    #     plt.imshow(net2.reshape(width,height,3))

    #     test_pic = np.zeros([1,flatten_num],dtype="float32")

    #     net3 = sess.run(net,feed_dict={input_x:test_pic})

    #     net3 = net3*255

    #     net3 = net3.astype("int")

    #     plt.figure(figsize=(12, 12))

    #     plt.imshow(net3.reshape(width,height,3))

    # compute test set loss
    # test_loss = 0
    # single_loss = 0
    # threshold = train_mean + 2 * train_std
    # print("threshold = ", threshold)
    # NG_num = 0;
    # good_num = 0
    # acc = 0
    # test_data = x_test_2
    # test_label = x_test_label_2
    #
    # for j in range(test_data.shape[0]):
    #
    #     single_loss = sess.run(loss, feed_dict={input_x: test_data[j:j + 1]})
    #
    #     if single_loss > threshold:  # predict NG
    #
    #         if test_label[j] == 0 or test_label[j] == 1:  # ground truth NG
    #
    #             acc += 1
    #         else:
    #             img = test_data[j] * 255
    #             img = img.astype("int")
    #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #             file_name = "FN" + str(j) + ".jpg"
    #             file_name = os.path.join(save_path, file_name)
    #             cv2.imwrite(file_name, img)
    #
    #
    #     elif single_loss >= 0:  # predict Good
    #
    #         if test_label[j] == 2:  # ground truth Good
    #
    #             acc += 1
    #         else:
    #             img = test_data[j] * 255
    #             img = img.astype("int")
    #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #             file_name = "FP" + str(j) + ".jpg"
    #             file_name = os.path.join(save_path, file_name)
    #             cv2.imwrite(file_name, img)
    #
    #     test_loss += single_loss
    #
    # test_loss /= test_data.shape[0]
    # acc /= test_data.shape[0]
    #
    # print('test_loss = ', test_loss)
    # print('accuracy = ', acc)
    #
    # # net1 = sess.run(net,feed_dict={input_x:x_test[0]})
    #
    # # net1.reshape(28,28)
    #
    # start = 0
    # index = 100
    #
    # plt.figure(figsize=(12, 12))
    # for i in range(start, start + 3, 1):
    #
    #     j = i + index
    #     plt.subplot(3, 3, 3 * i + 1)
    #     #         plt.imshow(x_test[j].reshape(width,height,3), vmin=0, vmax=1, cmap="brg")
    #     plt.imshow(test_data[j], vmin=0, vmax=1, cmap="brg")
    #     plt.title("Test input")
    #     plt.colorbar()
    #
    #     plt.subplot(3, 3, 3 * i + 2)
    #     prediction = sess.run(net, feed_dict={input_x: test_data[j:j + 1]})
    #     loss1 = sess.run(loss, feed_dict={input_x: test_data[j:j + 1]})
    #
    #     if loss1 > threshold:
    #         result = "NG"
    #     elif loss1 >= 0:
    #         result = "Good"
    #     print("Loss = {}\nLabel = {}".format(loss1, result))
    #     # plt.savefig(str(j)+result+".jpg")
    #
    #     diff = np.abs(prediction - test_data[j])
    #     # prediction = prediction*255
    #     # prediction = prediction.astype("int")
    #     plt.imshow(prediction.reshape(height, width, 3), vmin=0, vmax=1, cmap="brg")
    #     #         plt.imshow(net1, vmin=0, vmax=1, cmap="brg")
    #     plt.title("Reconstruction")
    #     plt.colorbar()
    #
    #     plt.subplot(3, 3, 3 * i + 3)
    #     #         diff = diff*255
    #     #         diff = diff.astype("int")
    #     plt.imshow(diff.reshape(height, width, 3), vmin=0, vmax=1, cmap="brg")
    #     #         plt.imshow(diff, vmin=0, vmax=1, cmap="brg")
    #     plt.title("Difference")
    #     plt.colorbar()
    #
    # plt.tight_layout()
