#coding=utf-8
import tensorflow as tf
import copy
import numpy as np
from texi import generateTexisPassengers,miniDist,bigramDist,arg2min
seed = 122
texis,passengers = generateTexisPassengers(seed=seed)
distance1 = miniDist(texis,passengers)
distance2,time_cost = bigramDist(texis,passengers)
tmp_texi = copy.deepcopy(texis)



# Network Parameters
n_hidden_1 = 8# 1st layer number of features
n_hidden_2 = 8 # 2nd layer number of features
n_input = 7 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with Tanh activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return tf.nn.softmax(out_layer)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, '../model/mlp.m')
    #calculate distance
    distance3 = 0
    for ele in passengers:
        ele_v = np.array(tmp_texi.shape[0]*[ele[0:2]])
        tds = np.linalg.norm(ele_v-tmp_texi,axis=1)
        try:
            ind1,ind2 = arg2min(tds)
        except:
            if len(tds) == 1:
                distance3 += np.linalg.norm(ele[0:2]-tmp_texi[0])
                break
        input_x = np.hstack((tmp_texi[ind1],tmp_texi[ind2],ele))
        input_x = np.array([input_x])
        y = sess.run(pred,feed_dict={x:input_x})
        p = np.argmax(y,axis=1)
        print p
        if p == 0:
            ind = ind1
        else:
            ind = ind2
        distance3 += np.linalg.norm(ele[0:2]-tmp_texi[ind])
        tmp_texi = np.delete(tmp_texi,ind,axis=0)
print 'Greedy:%f'%distance1
print 'MiniDistance:%f,time_cost:%f'%(distance2,time_cost)
print 'Mlp:%f'%distance3
