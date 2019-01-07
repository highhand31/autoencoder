import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import skimage
import common as cm
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model_filename = "model_saver/pb_test_model.pb"

# pic_path = r'E:\dataset\For_super_resolution\test'
pic_path = r'E:\dataset\face\test'
width = 256
height = 256
(x_ori, x_ori_label, no_care, no_care_2) = cm.data_load(pic_path, train_ratio=1,
                                                            normalize=False,has_dir=False,Only_addr=True)
print(x_ori.shape)
# x_noise = []
# for index in range(x_ori.shape[0]):
#     img = skimage.util.random_noise(x_ori[index], mode='gaussian', seed=None, clip=True)
#     #img = img *255
#     # img = float(img)
#     x_noise.append(img)
#     #print(index)
#
# x_noise = np.array(x_noise)
# x_noise = np.float32(x_noise)#執行完增噪後為float64，轉成float32，否則丟到input裡會出問題
# x_noise = int(x_noise)
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
with tf.Session() as sess:
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()

        tf.import_graph_def(graph_def, name='')  # 導入計算圖

    sess.run(tf.global_variables_initializer())

    ori_x = sess.graph.get_tensor_by_name("ori_x:0")
    supressed_x = sess.graph.get_tensor_by_name("supressed_x:0")
    # print(input_x.shape)
    result = sess.graph.get_tensor_by_name("output/Relu:0")
    print(result.shape)

    test_num = 12

    for num in range(test_num):
        print("pic path = ",x_ori[num])
        input_ori = []
        input_suppressed = []
        img = cv2.imread(x_ori[num])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        ori_img = cv2.resize(img, (256, 256))
        input_ori.append(ori_img)

        sup_img = cv2.resize(img,(128,128))
        input_suppressed.append(sup_img)


        input_ori = np.array(input_ori)
        input_ori = input_ori.astype("float32")
        input_ori = input_ori/255

        input_suppressed = np.array(input_suppressed)
        input_suppressed = input_suppressed.astype("float32")
        input_suppressed = input_suppressed / 255


        # input_x = x_ori[num:num+1]
        # input_x = x_ori[num:num + 1]
        reconstruct = sess.run(result, feed_dict={ori_x: input_ori,supressed_x:input_suppressed})
        # print(a)



        '''
        save pictures to compare
        '''
        out_path = r'G:\我的雲端硬碟\Python\Code\Pycharm\autoencoder\out\ori'
        #save ori
        img = cv2.imread(x_ori[num])
        addr_tail = "ori_"+x_ori[num].split("\\")[-1]
        total_path = os.path.join(out_path,addr_tail)
        print(total_path)
        #cv2.imwrite('.\out\ori_auditorium_00001342.jpg',img)#有中文路徑時會出現錯誤
        cv2.imencode('.jpg', img)[1].tofile(total_path) #有中文路徑時的寫入方法

        #save cv2 processed pic
        out_path = r'G:\我的雲端硬碟\Python\Code\Pycharm\autoencoder\out\cv2'
        img = cv2.resize(img,(128,128))
        img = cv2.resize(img, (256, 256))
        addr_tail = "cv2_" + x_ori[num].split("\\")[-1]
        total_path = os.path.join(out_path, addr_tail)
        cv2.imencode('.jpg',img)[-1].tofile(total_path)
        # cv2.imwrite(total_path, img)

        # save learning pic
        out_path = r'G:\我的雲端硬碟\Python\Code\Pycharm\autoencoder\out\ai'
        img = reconstruct[0]
        img = img *255
        # img = img.astype(np.int8)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        addr_tail = "AI_" + x_ori[num].split("\\")[-1]
        total_path = os.path.join(out_path, addr_tail)
        cv2.imencode('.jpg', img)[-1].tofile(total_path)



    '''
    display plots
    '''
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.imshow(reconstruct[0])
    #
    # plt.subplot(2,1,2)
    # plt.imshow(a[0])
    #
    # plt.show()