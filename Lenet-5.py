# Code from Sujay Babruwad extracted from https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb and modified
"""
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)

X_train_pre = mnist.train.images
y_train_pre = mnist.train.labels
X_train = X_train_pre[:25000]
y_train = y_train_pre[:25000]
X_validation = X_train_pre[25000:35000]
y_validation = y_train_pre[25000:35000]
X_test = X_train_pre[35000:]
y_test = y_train_pre[35000:]
"""

import itertools
import random
import numpy as np
from sklearn.utils import shuffle

num_classes = 9

data = np.genfromtxt("../../data/patch_data_storage/patch_data_all_rgb.csv", delimiter=",")
data = shuffle(data)

pure_list = []
for i in range(num_classes):
	pure_list.append([])

for i in range(len(data)):
	curr_label = int(data[i][len(data[i])-1])
	pure_list[curr_label].append(data[i])

for i in range(num_classes):
	print(len(pure_list[i]))


slice_num = [[1230, 300], [3016, 352], [3000, 300], [3000, 300], [3000, 300], [3000, 300], [3000, 300], [3000, 300], [3000, 300]]
#slice_num = [[120, 30], [150, 30], [150, 30], [150, 30], [150, 30], [150, 30], [150, 30], [150, 30], [150, 30]]

data_train = []
for i in range(num_classes):
	for j in range(slice_num[i][0]):
		data_train.append(pure_list[i][j])
data_train = np.asarray(data_train)
data_train = shuffle(data_train)
X_train = data_train[:,:-1]
y_train = data_train[:,-1:]
X_train = X_train.reshape([25246, 3, 28, 28])
X_train = X_train.transpose(0, 2, 3, 1)
y_train = y_train.reshape([25246, ])

data_test = []
for i in range(num_classes):
	for j in range(slice_num[i][1]):
		data_test.append(pure_list[i][slice_num[i][0] + j])
data_test = np.asarray(data_test)
data_test = shuffle(data_test)
X_test = data_test[:,:-1]
y_test = data_test[:,-1:]
X_test = X_test.reshape([2752, 3, 28, 28])
X_test = X_test.transpose(0, 2, 3, 1)
y_test = y_test.reshape([2752, ])

X_validation = X_test
y_validation = y_test


"""
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))
"""

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


# tps://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html
# Pad 2 pixels with edge values to length and width of images
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'edge')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'edge')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'edge')
    
print("Updated Image Shape: {}".format(X_train[0].shape))



import tensorflow as tf

EPOCHS = 13
BATCH_SIZE = 32


from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,3,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    """
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,10), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(10))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b

    logits = tf.matmul(fc1, fc2_w) + fc2_b
    """
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b

    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)



    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,48), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(48))
    fc3 = tf.matmul(fc2, fc3_w) + fc3_b

    fc3 = tf.nn.relu(fc3)

    fc4_w = tf.Variable(tf.truncated_normal(shape = (48, num_classes), mean = mu, stddev = sigma))
    fc4_b = tf.Variable(tf.zeros(num_classes))
    logits = tf.matmul(fc3, fc4_w) + fc4_b


    return logits

    """
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b

    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)


    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    return logits
    """

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, num_classes)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def calc_confusion_matrix(X_data, y_data):
	num_examples = len(X_data)

	CM = np.zeros((num_classes, num_classes), dtype=int)		# vetical: real, horizontal: predicted

	sess = tf.get_default_session()
	for offset in range(0, num_examples, BATCH_SIZE):
		batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
		predictions = sess.run(logits, feed_dict={x: batch_x})

		for i in range(BATCH_SIZE):
			#print(BATCH_SIZE)
			index = -1
			for j in range(num_examples):
				#print("i, j:   (" + str(i) + ", " + str(j) + ")")
				if np.isclose(predictions[i, j], predictions[i].max()):
					index = j
					break
			"""
			print("predictions[i]: ")
			print(predictions[i])
			print("j: ")
			print(j)
			print()

			print("batch_y[i]: ")
			print(batch_y[i])
			print()
			"""

			CM[int(batch_y[i]), index] += 1

	print(CM)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, "./MNIST.ckpt")
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    calc_confusion_matrix(X_test, y_test)

