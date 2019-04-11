
# coding: utf-8

# # MNIST CNN

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data


# In[4]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


# In[6]:


#HELPER

#INITIATE THE WEIGHTS

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


# In[8]:


#INITIATE THE BIAS VALUES
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


# In[10]:


#CONVENIENCE FUNCTION THAT HELPS CREATE THE 2D CONVOLUTION
#creating a wrapper that sets the parameters 
def conv2d(x,W):
    #x(Input Tensor) ---> [batch, H, W, Channels] 
    #W(Kernel)----> [Filter Heights, Filter W, Channels coming in, Channels Out ]
    
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')


# In[11]:


#POOLING
def max_pool_2by2(x):
    #x --> [batch, h, w, c]
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')


# In[18]:


#CONVOLUTIONAL LAYER
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


# In[19]:


#NORMAL LAYER(FULLY CONNECTED)
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# In[20]:


#PLACEHOLDERS
x = tf.placeholder(tf.float32, shape = [None, 784])


# In[21]:


y_true = tf.placeholder(tf.float32, shape = [None, 10])


# In[22]:


#LAYERS
x_image = tf.reshape(x, [-1, 28, 28, 1])


# In[23]:


convo_1 = convolutional_layer(x_image, shape = [5,5,1, 32])
convo_1_pooling = max_pool_2by2(convo_1)


# In[24]:


convo_2 = convolutional_layer(convo_1_pooling, shape= [5,5,32,64])
convo_2_pooling = max_pool_2by2(convo_2)


# In[25]:


#FLATTEN OUT
convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))


# In[30]:


# DROPOUT TO PREVENT OVERFITTING
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob= hold_prob)


# In[31]:


y_pred = normal_full_layer(full_one_dropout, 10)


# In[32]:


# LOSS FUNCTION
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred))


# In[33]:


#OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cross_entropy)


# In[34]:


init = tf.global_variables_initializer()


# In[36]:


steps = 5000

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train, feed_dict = {x:batch_x, y_true: batch_y, hold_prob:0.5})
        
        if i%100 == 0:
            print("ON STEP: {}".format(i))
            print("ACURACY: ")
            
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            print(sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels,hold_prob: 1.0}))
            print('\n')

