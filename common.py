# import tensorflow as tf
# from tensorflow.contrib import slim
import cv2
import os
import numpy as np
import matplotlib.image as mpimg
# import time
# import matplotlib.pyplot as plt


def data_load(pic_path, train_ratio=0.7, resize=None, shuffle=True, normalize=True, has_dir=True,Only_addr=False):
    labels = {}
    x_train = []
    x_train_label = []
    x_test = []
    x_test_label = []
    category_num = 1
    category_good = False
    num_jpg = 0
    num_png = 0
    num_others = 0
    format_jpg = {"jpg","JPG","jpeg","JPEG"}
    format_png = {"png","PNG"}


    if resize is not None:
        width = resize[0]
        height = resize[1]

        print('resize width = ', width)
        print('resize height = ', height)

    if train_ratio > 1:
        train_ratio = 1

    if has_dir is True:
        for num, dirs in enumerate(os.scandir(pic_path)):

            #確認是資料夾
            if dirs.is_dir():
                #確認資料夾名稱是否good，若有-->一律設定為0
                if dirs.name in {"Good","GOOD","good"}:
                    #print(dirs.name)
                    labels[dirs.name] = 0
                    category_good = True

                else:
                    labels[dirs.name] = category_num
                    category_num += 1
                #資料夾內的圖片處理
                file_path = os.path.join(pic_path, dirs.name)
                # print(file_path)
                files = [file.path for file in os.scandir(file_path) if file.is_file()]
                print("Picture number of dir({}) is {} ".format(file_path, len(files)))
                pic_length = len(files)

                train_num = int(pic_length * train_ratio)
                test_num = pic_length - train_num

                print('train data number = ', train_num)
                print('test data number = ', test_num)

                for pic_num, file in enumerate(files):

                    #要儲存成位址 或 圖片RGB內容
                    if Only_addr == False:#儲存圖片RGB內容

                        # img = cv2.imread(file)
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = mpimg.imread((file))
                        type_pic = file.split(".")[-1]

                        # 依照圖片格式計算數量
                        if type_pic in format_jpg:
                            num_jpg += 1
                        elif type_pic in format_png:
                            num_png += 1
                        else:
                            num_others += 1

                        # 圖片進行resize
                        if resize is not None:
                            img = cv2.resize(img, (width, height))
                        if pic_num < train_num:
                            x_train.append(img)
                            x_train_label.append(labels[dirs.name])
                        else:
                            x_test.append(img)
                            x_test_label.append(labels[dirs.name])
                    else:#儲存成位址
                        if pic_num < train_num:
                            x_train.append(file)
                            x_train_label.append(labels[dirs.name])
                        else:
                            x_test.append(file)
                            x_test_label.append(labels[dirs.name])


    else:#路徑內沒有資料夾，只有照片

        files = [file.path for file in os.scandir(pic_path) if file.is_file()]
        print("Picture number of dir({}) is {} ".format(pic_path, len(files)))
        pic_length = len(files)
        train_num = int(pic_length * train_ratio)
        test_num = pic_length - train_num
        print('train data number = ', train_num)
        print('test data number = ', test_num)

        for pic_num,file in enumerate(files):

            # 要儲存成位址 或 圖片RGB內容
            if Only_addr == False:  # 儲存圖片RGB內容
                #print(file)
                # img = cv2.imread(file)
                #print(img.shape)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = mpimg.imread((file))
                type_pic = file.split(".")[-1]

                #依照圖片格式計算數量
                if type_pic in format_jpg:
                    num_jpg +=1
                elif type_pic in format_png:
                    num_png +=1
                else:
                    num_others +=1


                # 圖片進行resize
                if resize is not None:
                    img = cv2.resize(img, (width, height))
                if pic_num < train_num:
                    x_train.append(img)
                    x_train_label.append(0)
                else:
                    x_test.append(img)
                    x_test_label.append(0)
                # flatten_num = img.shape[0]*img.shape[1]*img.shape[2]
                # img = img.reshape(flatten_num)
            else:  # 儲存成位址
                if pic_num < train_num:
                    x_train.append(file)
                    x_train_label.append(0)
                else:
                    x_test.append(file)
                    x_test_label.append(0)

    #-------------------

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train_label = np.array(x_train_label)
    x_test_label = np.array(x_test_label)

    #當執行多分類 且 分類裡沒有"good"時，分類的數字會從1開始，所以要減去1
    if has_dir == True and category_good == False:
        x_test_label -= 1


    # 將資料進行shuffle
    if shuffle is True:
        indice = np.random.permutation(x_train_label.shape[0])
        x_train = x_train[indice]
        x_train_label = x_train_label[indice]

        indice = np.random.permutation(x_test_label.shape[0])
        x_test = x_test[indice]
        x_test_label = x_test_label[indice]
        print("The data shuffle is done")
    # 將資料進行特徵縮放
    if normalize is True:
        #使用mpimg.imread((file))要注意，.png的預設會先進行normalization
        print("圖片中格式為jpg的數量 = {}，格式為png的數量 = {}，其他的數量 = {}".format(num_jpg, num_png, num_others))
        if num_png:
            print("圖片中含有png格式，已經進行normalization了，因此不再執行")

        else:
            x_train = x_train.astype("float32")
            x_train = x_train / 255

            x_test = x_test.astype("float32")
            x_test = x_test / 255
            print("The data normalization is done")

    print(labels)

    return (x_train, x_train_label, x_test, x_test_label)

if __name__ == "__main__":

    #for auto encoder classification
    #-->train data
    pic_path = r'E:\dataset\forAE\pill\train\Good'
    (x_train, x_train_label, no_care, no_care_2) = data_load(pic_path, train_ratio=1, resize=(64, 64),
                                                                       shuffle=True, normalize=True,has_dir=False)

    print('x_train shape = ', x_train.shape)
    print('x_train_label shape = ', x_train_label.shape)
    print(x_train_label)

    # -->test data
    #pic_path = r'E:\dataset\Surface_detection'
    pic_path = r'E:\dataset\forAE\pill\test'
    (no_care, no_care_2, x_test, x_test_label) = data_load(pic_path, train_ratio=0, resize=(64, 64),
                                                                          shuffle=True, normalize=True)
    # print('x_train shape = ',x_train_2.shape)
    # print('x_train_label shape = ',x_train_label_2.shape)
    print('x_test shape = ', x_test.shape)
    print('x_test_label shape = ', x_test_label.shape)
    print(np.max(x_test_label), np.min(x_test_label))
    #print("測試如果有中文打包的情形測試如果有中文打包的情形測試如果有中文打包的情形測試如果有中文打包的情形")

    # for multi classification
    pic_path = r'E:\dataset\flower_photos'
    (x_train_2, x_train_label_2, x_test_2, x_test_label_2) = data_load(pic_path, train_ratio=0.7, resize=(64, 64),
                                                           shuffle=True, normalize=True,
                                                               has_dir=True)

    print("For multi classification:\nx_train shape = {}\nx_train_label shape = {}".format(x_train_2.shape,x_train_label_2.shape))
    print("x_test shape = {}\nx_test_label shape = {}".format(x_test_2.shape,x_test_label_2.shape))
    print(np.max(x_test_label_2),np.min(x_test_label_2))

