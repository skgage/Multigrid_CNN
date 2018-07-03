#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:55:02 2018

@author: sarahgage
"""

#Run the following for tensorboard tensorboard --logdir=/tmp/summary/  --host=127.0.0.1 --port=6006

#--------Imports--------------------------------------
import tensorflow as tf
import numpy as np
import math
import mg_system_data
import matplotlib.pyplot as plt
tf.reset_default_graph()
import sys

#-----------------------------------------------------
#-----------------------------------------------------

#WORTH CLIPPING X AND INVERSE VALUES?

#-------------Global Variables ----------------------

learning_rate = 1e-4
level = 2
level_prime = 0
batch_size = 10
num_data= 100
gridsize =  8
epoch_num = 100
num_batch = 1
dim = 1 #number of dimensions 
mean = np.reshape([0.5, 1, 0.5], [1,3,1,1])
sigma = 0.0
equation = 'laplacian'
#----------------------------------------------------
#-----------------------------------------------------


#--------------Model ----------------------------------------------
def conv_restrict(input_tensor, level, level_prime):
    batch_size = tf.shape(input_tensor)[0]
    size = int(gridsize/(2**level_prime))
    #size = tf.shape(input_tensor)[2]
    a  = tf.gather(input_tensor, [size-1], axis=2)
    input_tensor = tf.concat([a,input_tensor], axis=2)
    a = tf.random_normal((1,3,1,1),mean, sigma)
    #print ('a = {}'.format(a))
    init = a/(tf.norm(a)/sigma)
    #init = tf.constant((a/(np.linalg.norm(a)/sigma)),dtype=tf.float32)
    #init = np.random.normal(mean, sigma, size=(1,3,1,1))
    #weights = tf.get_variable(name='R{}'.format(level),shape=[1,3,1,1], initializer=init) 
    #b = sigma*np.random.rand()+0.5
    #a = tf.Variable(b, name='Re{}'.format(level),trainable=True)
    #init = tf.reshape([a, 1, a], [1,3,1,1])
    #with tf.variable_scope('scope', reuse = tf.AUTO_REUSE ):
    weights = tf.get_variable(name='R{}'.format(level),initializer=init, trainable=True) 
    conv = tf.nn.conv2d(input_tensor, weights, strides=[1,1,2,1], padding="VALID")
    conv = tf.reshape(conv, [batch_size,1, int(size/2), 1])
    return conv


def conv_prolong(input_tensor, level, level_prime):
    batch_size = tf.shape(input_tensor)[0]
    size = int(gridsize/(2**level_prime))
    #size = tf.shape(input_tensor)[2]
    #b = sigma*np.random.rand()+0.5
    #a = tf.Variable(b, name='Pr{}'.format(level),trainable=True)
    #init = tf.reshape([a, 1, a], [1,3,1,1])
    #a = np.random.normal(mean, sigma, size=(1,3,1,1))
    a = tf.random_normal((1,3,1,1),mean, sigma)
    init = a/(tf.norm(a)/sigma)
    #init = tf.constant((a/(np.linalg.norm(a)/sigma)),dtype=tf.float32)
    #with tf.variable_scope('scope', reuse = tf.AUTO_REUSE ):
    weights = tf.get_variable(name = 'P{}'.format(level), initializer=init, trainable=True) 
    conv = tf.nn.conv2d_transpose(input_tensor, weights, 
                                  output_shape=[batch_size,1,size+1,1], strides=[1,1,2,1], padding="VALID")
    iconv = conv[:,:,1:,:] + tf.pad(conv[:,:,0:1,:], [(0,0), (0,0), (size-1,0), (0,0)])
    return tf.reshape(iconv, (batch_size,1,size,1))

def inverse_layer(input_tensor, level, level_prime):
    size = tf.shape(input_tensor)[2]
    #mean = np.linalg.pinv(Laplacian(size))
    mean = np.array([[ 0.5, -0.5],[-0.5,  0.5]])*(2**(level_prime-2))
    #a = np.random.normal(mean, sigma, size=(2,2))
    #init = tf.constant((a/(np.linalg.norm(a)/sigma)),dtype=tf.float32)
    a = tf.random_normal((2,2),mean, sigma)
    init = a/(tf.norm(a)/sigma)
    #init = tf.constant(np.random.normal(mean, sigma, size=(2,2)),dtype=tf.float32)
    #init = mean.astype(np.float32)
    #with tf.variable_scope('scope', reuse = tf.AUTO_REUSE ):
    weights = tf.get_variable(name='Inverse_level{}'.format(level),initializer=init, validate_shape=False)
    input_tensor = tf.reshape(input_tensor, [tf.shape(input_tensor)[0], size])
    inv = tf.reshape(tf.matmul(input_tensor,weights), [tf.shape(input_tensor)[0], 1,size, 1])
    return inv

def diagonal(input_tensor, level, level_prime):
    batch_size = tf.shape(input_tensor)[0]
    size = int(gridsize/(2**(level_prime+1)))
    #size = tf.shape(input_tensor)[2]
    D = tf.tile([0,1],[size],name='D{}'.format(level))
    D = tf.cast(D, tf.float32)
    #with tf.variable_scope('scope', reuse = tf.AUTO_REUSE ):
    c = tf.Variable(0.5*2**(level_prime),name='x{}'.format(level), trainable=True)
        #c = tf.clip_by_value(c, 0, 50)
    D = c*D
    D = tf.expand_dims(tf.expand_dims(tf.expand_dims(D, 1), 0),0)
    Db = D * input_tensor
    return Db #this shape should be (1,1,8,1)

def output_layer(tensor):
    size = tf.shape(tensor)[0]
    return tensor - tf.reshape(tf.reduce_mean(tensor,2), [size,1,1,1])

def model(b, level, level_prime):
   # tf.reset_default_graph()
    if level == 0:
        print ('hello')
        return inverse_layer(b, level, level_prime)
    else:
        if (b.shape[2] % 2 == 1):
            raise RuntimeError('Cannot restrict from level {} with shape {}'.format(level, b.shape[2]))
        Db = diagonal(b, level, level_prime)
        Rb = conv_restrict(b, level, level_prime)
        bc = model(Rb, level-1, level_prime+1)
        Mb = conv_prolong(bc, level, level_prime)
        return output_layer(Db + Mb)
#----------------------------------------------------------
#----------------------------------------------------------
        

#--------------Hessian computation --------------------------------------
def condense_hessian(h_vars):
    v = []
    for var in h_vars:
        v.append(tf.squeeze(var))
    return v

def cond_num(h_vars):
    v = []
    for var in h_vars:
        if var.ndim <= 1:
            continue
        elif var.ndim == 4:
            var = np.reshape(var, [4,4])
        #print ('var shape = {}'.format(np.shape(var)))
        cond = np.linalg.cond(var)
        if cond > 1000:
            print('unconditioned var = {}'.format(var))
        v.append(cond)
    return v
#----------------------------------------------------------
  
    
#---------------Placeholders and Function calls --------------------------
#tf.reset_default_graph()
#with tf.variable_scope('scope', reuse = tf.AUTO_REUSE ):
b = tf.placeholder("float", shape=[None,1,gridsize,1], name='b')
u_ = tf.placeholder("float", shape=[None,1, gridsize,1], name='u')

learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
output_u_ = model(b, level, level_prime)
output = tf.Print(output_u_, [output_u_])

loss = tf.losses.mean_squared_error(output_u_, u_)

#hessian = tf.hessians(loss, tf.trainable_variables())
#hessian = condense_hessian(hessian)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_placeholder)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_placeholder, momentum=0.9, use_nesterov=True)
train = optimizer.minimize(loss)

correct = tf.equal(output_u_, u_)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



def network_training():
    #tf.reset_default_graph()
    init = tf.global_variables_initializer()
    #saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()
    sess.run(init)
    training_loss_plt = [0]*(epoch_num*num_batch)
    testing_loss_plt = [0]*(epoch_num*num_batch)
    epoch = [0]*(epoch_num*num_batch)
    shift = 0
    for n in range(num_batch):
        b_vals, actual_u_outputs = mg_system_data.gen_data(gridsize, int(num_data*4/5), dim, equation, 0)
        test_b_vals, test_actual_u_outputs = mg_system_data.gen_data(gridsize, int(num_data), dim, equation, num_data)
        #print ('b_vals {}, training actual_u_outputs {}, test_b_vals {}, test_actual_u_outputs {}'.format(b_vals, actual_u_outputs, test_b_vals, test_actual_u_outputs))
        #sess.run(reset, {b: b_vals})
        init_vars = {}
        i=1
        for var in tf.trainable_variables():
            var_val = sess.run(var)
            init_vars['{}'.format(var.name)]= var_val
            #print(np.shape(var_val))
            i+=1
        error_prime = np.inf
        learning_rate_prime = learning_rate
        for e in range(epoch_num):
            #learning_rate = 1e-4 if e < 10 else 1e-2
            for i in range(int(num_data/batch_size)):
                batch_x = b_vals[i:i+batch_size,:,:]
                batch_y = actual_u_outputs[i:i+batch_size,:,:]
                sess.run(train, {b: batch_x.astype(np.float32), u_: batch_y.astype(np.float32), learning_rate_placeholder: learning_rate_prime})
                #sess.run(clip_op)
            training_error = sess.run(loss, {b: np.array(b_vals), u_: np.array(actual_u_outputs), learning_rate_placeholder: learning_rate_prime})
            testing_error = sess.run(loss, {b: np.array(test_b_vals), u_: np.array(test_actual_u_outputs), learning_rate_placeholder: learning_rate_prime})
            print('epoch # ', e+shift, 'training_error = {}, testing_error = {}'.format(training_error, testing_error))
            if math.isnan(training_error):
                break
            #print ('output_u_', sess.run(output_u_, {b: np.array(b_vals), u_: np.array(actual_u_outputs), learning_rate_placeholder: learning_rate}))
            training_loss_plt[e+shift] = training_error
            testing_loss_plt[e+shift] = testing_error
            epoch[e+shift] = e+shift
#            learning_rate_prime = learning_rate if (error_prime < training_error or training_error > 40000000) else (learning_rate*100)
#            learning_rate_prime = learning_rate_prime if (error_prime < training_error or training_error > 10000000) else (learning_rate_prime*100)
#            learning_rate_prime = learning_rate_prime if (error_prime < training_error or training_error > 50000) else (learning_rate_prime*100)
#            learning_rate_prime = learning_rate_prime if (error_prime < training_error or training_error > 3) else (learning_rate_prime*1000)
#            learning_rate_prime = learning_rate_prime if (error_prime < training_error or training_error > 1) else (learning_rate_prime*100)
#            learning_rate_prime = learning_rate_prime if (error_prime < training_error or training_error > 0.06) else (learning_rate_prime*10)
            #learning_rate_prime = learning_rate_prime if (error_prime < error or error > 0.035) else (learning_rate_prime*5)
            error_prime = training_error
            
            #hess = sess.run(hessian, {b: batch_x.astype(np.float32), u_: batch_y.astype(np.float32), learning_rate_placeholder: learning_rate_prime})
            #hess_cond_num = cond_num(hess)
            #print("hessian = %s" % hess_cond_num)
            
        #save_path = saver.save(sess, "/tmp/model.ckpt")
        #print("Model saved in path: %s" % save_path)
        converged_vars = {}
        #converged_u = sess.run(output_u_, {b: np.array(b_vals), u_: np.array(actual_u_outputs)})
        j=i+1
        for var in tf.trainable_variables():
            var_val = sess.run(var)
            #sh.write(trial+1,j,'{}'.format(var_val))
            #print ('var name = ', var.name,' values = ', var_val)
            converged_vars['{}'.format(var.name)]= var_val
            j+=1
        #print ('initial params = {}'.format(init_vars))
        #print ('converged params = {}'.format(converged_vars))
        #print('epoch # ', e+shift, 'training_error =', error)
    plt.plot(epoch,training_loss_plt,label='training error')
    plt.plot(epoch, testing_loss_plt,label='validation error')
    plt.title("Loss Plot")
    plt.xlabel("Cumulative Epoch")
    plt.ylabel("Mean squared error")
    plt.yscale('log')
    #plt.legend()
    #plt.show()
    sess.close()
    return training_error #converged_vars, init_vars, epoch, training_loss_plt 


network_training()
