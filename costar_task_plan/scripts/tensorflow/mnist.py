#!/usr/bin/env python

from tensorflow.examples.tutorials.mnist import input_data

# mnist stores all of the training, test, and validation data as NumPy arrays
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print "Test examples = %d"%(mnist.test.num_examples)

# a session is a c++ backend for tensorflow computation
# 
import tensorflow as tf
sess = tf.InteractiveSession()

'''
Setting up a basic regression model
'''

# x is the input images -- they are 784 dimension binary vectors
x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ is the correct label of the training data -- digits 0 through 0, so 10 dimensional
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# the first dimension of the shape is None because these can be of any size (amount of training data)

# now we can create our weight matrices and biases for each class
# bias is the prior preference for each digit
# weight is input image dimensionality x number of digits
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# initialize all variables to be full of zeros
sess.run(tf.initialize_all_variables())

# set up the actual regression model
y = tf.nn.softmax(tf.matmul(x,W) + b)

# set up the cost function we want to optimize
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# this sums across all classes, then takes the mean of these sums to result in one score
# we need this because y is 10 x ??? dimensional...

# now we actually set up our training optimization operation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# steepest gradient descent with step size of 0.5

# now let's just run for 100 iterations and see how it does
for i in range(1000):
    batch = mnist.train.next_batch(50)
    # this loads 50 training examples as a tuple (x, y)
    # (x, y) = mnist.train.next_batch(50) returns:
    # x is a 50 x 784 matrix of flattened MNIST images
    # y is a 50 x 10 matrix of one-hot MNIST labels

    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# check to see if our final outputs are the correct ones --
# argmax chooses the highest value along a particular axis
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# use tensorflow.cast to convert booleans to floating point numbers
# then take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# evaluate the accuracy expression we just created
# set x to all image data
# set y to all label data
print "Simple regression model accuracy: %f"%(
        accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

'''
Setting up a simple multilayer convolutional neural net
'''

# We will now need to create a lot of weights and biases
# We want to initialize these randomly with a small positive bias
# Avoid zeros so we don't have a zero gradient
# A slight positive bias helps prevent "dead neurons"
# we will make two functions to do this:

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# next we want to choose our convolution and pooling sizes
# this is the kind of thing we can play around with to improve performance

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# padding and stride size --
# zero padded so dimensionality is the same

# creating weights for the conv layers
# and bias for the conv layers
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 5 x 5 input field
# computes 32 features for each 5 x 5 patch
# one input channel

# we need to reshape x into a 4d tensor
# second and third dimensions are image height and width
# final dimension is number of color channels
x_image = tf.reshape(x, [-1,28,28,1])

# finally we can actually create a convolutional layer
# we convolve x image with weights, add biases, and finally add ReLU
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# then we do our pooling
h_pool1 = max_pool_2x2(h_conv1)


# now we do another one --
# 32 inputs
# 64 features per 5 x 5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# now this has been reduced from a 28 x 28 to a 7 x 7 image thanks to our two pooling iterations
# we want a fully connected layer on these two pooling levels
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
# this is supposed to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# set up fully connected weights and biases
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# finally add another softmax
# this is the softmax
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
#for i in range(20000):
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

test_size = 1000
batch_size = 50
acc = 0
nsteps = test_size / batch_size
for step in xrange(nsteps):
    offset = step * batch_size
    batch_data = mnist.test.images[offset:(offset + batch_size),:]
    batch_labels = mnist.test.labels[offset:(offset+batch_size)]
    feed_dict = {x: batch_data, y_: batch_labels, keep_prob: 1.0}
    acc += accuracy.eval(feed_dict=feed_dict) /  nsteps # really lazy mean of means
    
print("test accuracy %g"%acc)
#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


'''
step 0, training accuracy 0.14
step 100, training accuracy 0.86
step 200, training accuracy 0.88
step 300, training accuracy 0.96
step 400, training accuracy 0.94
step 500, training accuracy 0.92
step 600, training accuracy 0.94
step 700, training accuracy 0.94
step 800, training accuracy 0.9
step 900, training accuracy 0.96
step 1000, training accuracy 0.96
step 1100, training accuracy 0.88
step 1200, training accuracy 0.94
step 1300, training accuracy 1
step 1400, training accuracy 0.98
step 1500, training accuracy 1
step 1600, training accuracy 0.98
step 1700, training accuracy 0.94
step 1800, training accuracy 0.96
step 1900, training accuracy 1
step 2000, training accuracy 0.96
step 2100, training accuracy 1
step 2200, training accuracy 0.92
step 2300, training accuracy 1
step 2400, training accuracy 0.96
step 2500, training accuracy 0.98
step 2600, training accuracy 0.96
step 2700, training accuracy 1
step 2800, training accuracy 0.98
step 2900, training accuracy 0.96
step 3000, training accuracy 1
step 3100, training accuracy 0.98
step 3200, training accuracy 0.96
step 3300, training accuracy 1
step 3400, training accuracy 1
step 3500, training accuracy 1
step 3600, training accuracy 0.98
step 3700, training accuracy 0.94
step 3800, training accuracy 0.98
step 3900, training accuracy 0.98
step 4000, training accuracy 0.96
step 4100, training accuracy 0.98
step 4200, training accuracy 1
step 4300, training accuracy 1
step 4400, training accuracy 0.98
step 4500, training accuracy 0.98
step 4600, training accuracy 1
step 4700, training accuracy 1
step 4800, training accuracy 1
step 4900, training accuracy 0.98
step 5000, training accuracy 1
step 5100, training accuracy 0.98
step 5200, training accuracy 1
step 5300, training accuracy 1
step 5400, training accuracy 0.96
step 5500, training accuracy 0.96
step 5600, training accuracy 0.96
step 5700, training accuracy 1
step 5800, training accuracy 1
step 5900, training accuracy 1
step 6000, training accuracy 1
step 6100, training accuracy 0.98
step 6200, training accuracy 1
step 6300, training accuracy 1
step 6400, training accuracy 0.98
step 6500, training accuracy 0.98
step 6600, training accuracy 1
step 6700, training accuracy 0.98
step 6800, training accuracy 0.98
step 6900, training accuracy 1
step 7000, training accuracy 0.96
step 7100, training accuracy 1
step 7200, training accuracy 1
step 7300, training accuracy 0.98
step 7400, training accuracy 0.98
step 7500, training accuracy 0.96
step 7600, training accuracy 1
step 7700, training accuracy 1
step 7800, training accuracy 1
step 7900, training accuracy 1
step 8000, training accuracy 0.96
step 8100, training accuracy 1
step 8200, training accuracy 1
step 8300, training accuracy 0.98
step 8400, training accuracy 0.98
step 8500, training accuracy 1
step 8600, training accuracy 1
step 8700, training accuracy 1
step 8800, training accuracy 0.98
step 8900, training accuracy 1
step 9000, training accuracy 1
step 9100, training accuracy 1
step 9200, training accuracy 1
step 9300, training accuracy 0.98
step 9400, training accuracy 1
step 9500, training accuracy 0.98
step 9600, training accuracy 1
step 9700, training accuracy 1
step 9800, training accuracy 1
step 9900, training accuracy 0.98
step 10000, training accuracy 1
step 10100, training accuracy 1
step 10200, training accuracy 0.96
step 10300, training accuracy 1
step 10400, training accuracy 0.98
step 10500, training accuracy 0.98
step 10600, training accuracy 0.98
step 10700, training accuracy 1
step 10800, training accuracy 1
step 10900, training accuracy 1
step 11000, training accuracy 0.96
step 11100, training accuracy 1
step 11200, training accuracy 1
step 11300, training accuracy 1
step 11400, training accuracy 0.96
step 11500, training accuracy 1
step 11600, training accuracy 1
step 11700, training accuracy 1
step 11800, training accuracy 1
step 11900, training accuracy 1
step 12000, training accuracy 1
step 12100, training accuracy 1
step 12200, training accuracy 0.98
step 12300, training accuracy 1
step 12400, training accuracy 1
step 12500, training accuracy 1
step 12600, training accuracy 1
step 12700, training accuracy 0.98
step 12800, training accuracy 1
step 12900, training accuracy 0.98
step 13000, training accuracy 1
step 13100, training accuracy 1
step 13200, training accuracy 1
step 13300, training accuracy 0.96
step 13400, training accuracy 1
step 13500, training accuracy 1
step 13600, training accuracy 1
step 13700, training accuracy 1
step 13800, training accuracy 1
step 13900, training accuracy 1
step 14000, training accuracy 1
step 14100, training accuracy 1
step 14200, training accuracy 1
step 14300, training accuracy 1
step 14400, training accuracy 1
step 14500, training accuracy 1
step 14600, training accuracy 1
step 14700, training accuracy 1
step 14800, training accuracy 1
step 14900, training accuracy 1
step 15000, training accuracy 1
step 15100, training accuracy 1
step 15200, training accuracy 1
step 15300, training accuracy 1
step 15400, training accuracy 1
step 15500, training accuracy 1
step 15600, training accuracy 1
step 15700, training accuracy 1
step 15800, training accuracy 1
step 15900, training accuracy 1
step 16000, training accuracy 1
step 16100, training accuracy 1
step 16200, training accuracy 0.98
step 16300, training accuracy 1
step 16400, training accuracy 1
step 16500, training accuracy 1
step 16600, training accuracy 1
step 16700, training accuracy 1
step 16800, training accuracy 1
step 16900, training accuracy 1
step 17000, training accuracy 1
step 17100, training accuracy 1
step 17200, training accuracy 0.98
step 17300, training accuracy 1
step 17400, training accuracy 1
step 17500, training accuracy 1
step 17600, training accuracy 0.98
step 17700, training accuracy 1
step 17800, training accuracy 1
step 17900, training accuracy 1
step 18000, training accuracy 1
step 18100, training accuracy 1
step 18200, training accuracy 1
step 18300, training accuracy 1
step 18400, training accuracy 0.98
step 18500, training accuracy 1
step 18600, training accuracy 1
step 18700, training accuracy 1
step 18800, training accuracy 1
step 18900, training accuracy 1
step 19000, training accuracy 1
step 19100, training accuracy 1
step 19200, training accuracy 1
step 19300, training accuracy 1
step 19400, training accuracy 1
step 19500, training accuracy 1
step 19600, training accuracy 0.98
step 19700, training accuracy 1
step 19800, training accuracy 1
step 19900, training accuracy 1
test accuracy 0.9921
'''
