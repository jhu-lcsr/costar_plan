#!/usr/bin/env python

import task_tree_search as tts
import tensorflow as tf
import numpy as np
import random # used to sample from memory

# take this training data and learn a neural net saying which actions are likely
with open('train.npz','r') as infile:
    npz = np.load(infile)
    xdata = npz['x']
    ydata = npz['y']

rdata = np.ones((xdata.shape[0],1))

num_features = xdata.shape[1]
num_actions = ydata.shape[1]

sess = tf.InteractiveSession()

# now we can create our weight matrices and biases for each class
# bias is the prior preference for each action
# weight is input dimensionality x number of actions
W0 = tf.Variable(tf.truncated_normal([num_features,num_features], stddev=0.1))
b0 = tf.Variable(tf.constant(0.1, shape=[num_features]))
W1 = tf.Variable(tf.truncated_normal([num_features,num_actions], stddev=0.1))
b1 = tf.Variable(tf.zeros(shape=[num_actions]))

# start with a simple regression model instead of a neural net
state = tf.placeholder(tf.float32, shape=[None, num_features])
reward = tf.placeholder(tf.float32, shape=[None, 1])
action = tf.placeholder(tf.float32, shape=[None, num_actions])

# set up the actual regression model
y0 = tf.nn.relu(tf.matmul(state,W0) + b0)
y = tf.nn.softmax(tf.matmul(y0,W1) + b1)
predicted_reward = tf.reduce_sum(tf.mul(y,action), reduction_indices=1)

# set up the cost function we want to optimize
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
loss = tf.reduce_mean(tf.square(reward - predicted_reward))

# now we actually set up our training optimization operation
#train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# check to see if our final outputs are the correct ones --
# argmax chooses the highest value along a particular axis
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(action,1))

# use tensorflow.cast to convert booleans to floating point numbers
# then take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize all variables to be full of zeros
sess.run(tf.initialize_all_variables())

# set up the feature function we want to use to represent worlds
def get_features(world,actor):
    nw = world.getLocalWorld(actor)
    features = nw.getFeatures(actor,useIntersection=True,flattened=True)
    return features

exp = tts.Experience(30000,num_features,num_actions,discount=1)
exp.initFromArrays(xdata,ydata)
exp.setModel(model_input=state,model_output=y)

# train using all data for 1000 steps
for i in xrange(300):
    (xin,yin,rin) = exp.sample(2000)
    train_step.run(feed_dict={state: xin, action: yin, reward: rin})
    if i % 100 == 0:
        print "-- iter %d accuracy = %f"%(i,accuracy.eval(feed_dict={state: xdata, action: ydata}))

# new memory -- only use newly acquired training data right now
# because we're training that differently!
#exp = tts.Experience(30000,num_features,num_actions,discount=0)
exp.setModel(model_input=state,model_output=y)

for i in xrange(5):

    tts.RunQlearningIter(
            x_var = state,
            y_var = action,
            r_var = reward,
            x_dim = num_features,
            y_dim = num_actions,
            y_out = y,
            train_step = train_step,
            get_world = tts.GetCrossworldForIter,
            get_features = get_features,
            experience = exp,
            evaluate_world = tts.EvaluateAndGetFeatures,
            accuracy = accuracy,
            valid_x_data = xdata,
            valid_y_data = ydata,
            num_simulations = 100,
            retrain_iter = 101,
            index = 0,
            r_out = predicted_reward
            )

(world, actor) = tts.GetCrossworldForIter(state,num_features,y)

'''
END
'''

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

sess.close()
