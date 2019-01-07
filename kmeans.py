import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import pandas as pd
from pandas import ExcelWriter

# num_points = 55000
# dimensions = 28
# points = np.random.uniform(0, 1, [num_points, dimensions,dimensions])

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
input_mnist = mnist.train.images[:50]
label_mnist = mnist.train.labels[:50]
# input_mnist = np.reshape(input_mnist,(-1,28,28))#只能夠丟進去2維的矩陣
print(input_mnist.shape)
print("label mnist shape = ",label_mnist.shape)

def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(input_mnist, dtype=tf.float32), num_epochs=1)

num_clusters = 10
kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=num_clusters, use_mini_batch=True)


# train
num_iterations = 3
previous_centers = None
for _ in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  if previous_centers is not None:
    print('delta:', cluster_centers - previous_centers)
  previous_centers = cluster_centers
  print ('score:', kmeans.score(input_fn))
print ('cluster centers:', cluster_centers)

# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
cluster_indices = np.array(cluster_indices)
print('cluster_indices shape = ',cluster_indices.shape)

writer = ExcelWriter('confusion_test.xlsx')
plot = pd.crosstab(label_mnist,cluster_indices,rownames=['label'],colnames=['K'])
plot.to_excel(writer, 'Sheet1')
writer.save()
# for i, point in enumerate(input_mnist):
#   cluster_index = cluster_indices[i]
#   center = cluster_centers[cluster_index]
#   print ('point:', i, 'is in cluster', cluster_index, 'centered at', center)