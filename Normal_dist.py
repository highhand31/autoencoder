import tensorflow as tf
from tensorflow import distributions as tfd
# tfd = tfp.distributions

# Define a single scalar Normal distribution.
dist = tfd.Normal(loc=0., scale=1.)#loc 表示平均值，scale表示標準差
# print(dist)

# Evaluate the cdf at 1, returning a scalar.
# print(dist.cdf(1.))

# Define a batch of two scalar valued Normals.
# The first has mean 1 and standard deviation 11, the second 2 and 22.
# dist = tfd.Normal(loc=[1, 2.], scale=[11, 22.])
#
# # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
# # returning a length two tensor.
# dist.prob([0, 1.5])
#
# # Get 3 samples, returning a 3 x 2 tensor.
# dist.sample([3])

# 設定GPU參數
config = tf.ConfigProto(log_device_placement=True,
                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                        )
config.gpu_options.per_process_gpu_memory_fraction = 0.1

with tf.Session(config=config) as sess:

    print(sess.run(dist.cdf(1.)))

    print(sess.run(dist.prob(0.)))

    print(sess.run(dist.sample(3)))