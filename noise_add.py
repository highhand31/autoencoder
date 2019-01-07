'''
ref:
https://www.smwenku.com/a/5b87615d2b71775d1cd6d55b/

'''
import common as cm
import skimage
import matplotlib.pyplot as plt
import numpy as np




pic_path = r'E:\dataset\forAE\circle\train\Good'
# pic_path = r'E:\dataset\JK\out_pic'
(x_train, x_train_label, no_care, no_care_2) = cm.data_load(pic_path, train_ratio=1, has_dir=False,normalize=False)
print(x_train[0].shape)
print(x_train[0])
ori = x_train[0]
print(ori.dtype)

'''
mode : str
    One of the following strings, selecting the type of noise to add:
    ‘gaussian’ Gaussian-distributed additive noise.
    ‘localvar’ Gaussian-distributed additive noise, with specified local variance at each point of image
    ‘poisson’ Poisson-distributed noise generated from the data.
    ‘salt’ Replaces random pixels with 1.
    ‘pepper’ Replaces random pixels with 0 (for unsigned images) or -1 (for signed images).
    ‘s&p’ Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signedimages.
    ‘speckle’ Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.
'''

#add noise
img = skimage.util.random_noise(ori, mode='gaussian', seed=None, clip=True)
img_2 = skimage.util.random_noise(ori, mode='salt', seed=None, clip=True)
img_3 = skimage.util.random_noise(ori, mode='pepper', seed=None, clip=True)
print(img.dtype)



plot = plt.figure()
plt.subplot(2,2,1)
plt.imshow(x_train[0])

plt.subplot(2,2,2)
plt.imshow(img)

plt.subplot(2,2,3)
plt.imshow(img_2)

plt.subplot(2,2,4)
plt.imshow(img_3)

plt.show()
