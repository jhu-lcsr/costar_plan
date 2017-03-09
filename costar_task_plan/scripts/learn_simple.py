#!/usr/bin/env python

import task_tree_search as tts
import tensorflow as tf
import numpy as np

# take this training data and learn a neural net saying which actions are likely
with open('train.npz','r') as infile:
    npz = np.load(infile)
    xdata = npz['x']
    ydata = npz['y']

print xdata
print ydata

num_features = xdata.shape[1]
num_actions = ydata.shape[1]

sess = tf.InteractiveSession()

# now we can create our weight matrices and biases for each class
# bias is the prior preference for each action
# weight is input dimensionality x number of actions
W = tf.Variable(tf.zeros([num_features,num_actions]))
b = tf.Variable(tf.zeros([num_actions]))

# initialize all variables to be full of zeros
sess.run(tf.initialize_all_variables())

# start with a simple regression model instead of a neural net
x = tf.placeholder(tf.float32, shape=[None, num_features])
y_ = tf.placeholder(tf.float32, shape=[None, num_actions])

# set up the actual regression model
y = tf.nn.softmax(tf.matmul(x,W) + b)

# set up the cost function we want to optimize
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# now we actually set up our training optimization operation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# check to see if our final outputs are the correct ones --
# argmax chooses the highest value along a particular axis
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# use tensorflow.cast to convert booleans to floating point numbers
# then take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train using all data for 1000 steps
for i in xrange(1000):
    train_step.run(feed_dict={x: xdata, y_: ydata})
    if i % 100 == 0:
        print "-- iter %d accuracy = %f"%(i,accuracy.eval(feed_dict={x: xdata, y_: ydata}))

print "Training accuracy = %f"%(accuracy.eval(feed_dict={x: xdata, y_: ydata}))

# we can easily get our model out with y
#y.eval(feed_dict={x: xdata, y_:ydata})

'''
Create a random world and see how our actor does!
'''

world = tts.GetCrossworld()
for i in range(10):
    world.addActor(tts.GetCrossworldDefaultActor(world,str(i)))
tmp = tts.GetCrossworldDefaultActor(world,"A")
reg = tts.RegressionActor(tmp.state)
reg.setModel(y,x,num_features)
world.addActor(reg)

tg = tts.TerminalGraphics()
tg.drawWorld(world)
try:
    while True:
        tg.wait()
        world.tick()
        tg.drawWorld(world)
except KeyboardInterrupt, e:
    pass
finally:
    tg.close()
    sess.close()
