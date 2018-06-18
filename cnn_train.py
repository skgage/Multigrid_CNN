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
import hessian_comp
tf.reset_default_graph()
#-----------------------------------------------------
#-----------------------------------------------------



#-------------Global Variables ----------------------
learning_rate = 1e-10
level = 7
level_prime = 0
batch_size = 16
num_data = 256
gridsize =  256
epoch_num = 10
num_batch = 1
dim = 1 #number of dimensions 
mean = np.reshape([0.5, 1, 0.5], [1,3,1,1])
sigma = 1.0
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
    init = tf.constant(np.random.normal(mean, sigma, size=(1,3,1,1)),dtype=tf.float32)
    #init = np.random.normal(mean, sigma, size=(1,3,1,1))
    #weights = tf.get_variable(name='R{}'.format(level),shape=[1,3,1,1], initializer=init) 
    weights = tf.get_variable(name='R{}'.format(level),initializer=init, trainable=True) 
    conv = tf.nn.conv2d(input_tensor, weights, strides=[1,1,2,1], padding="VALID")
    conv = tf.reshape(conv, [batch_size,1, int(size/2), 1])
    return conv


def conv_prolong(input_tensor, level, level_prime):
    batch_size = tf.shape(input_tensor)[0]
    size = int(gridsize/(2**level_prime))
    #size = tf.shape(input_tensor)[2]
    init = tf.constant(np.random.normal(mean, sigma, size=(1,3,1,1)),dtype=tf.float32)
    weights = tf.get_variable(name = 'P{}'.format(level), initializer=init, trainable=True)  
    conv = tf.nn.conv2d_transpose(input_tensor, weights, 
                                  output_shape=[batch_size,1,size+1,1], strides=[1,1,2,1], padding="VALID")
    iconv = conv[:,:,1:,:] + tf.pad(conv[:,:,0:1,:], [(0,0), (0,0), (size-1,0), (0,0)])
    return tf.reshape(iconv, (batch_size,1,size,1))

def inverse_layer(input_tensor, level, level_prime):
    size = tf.shape(input_tensor)[2]
    #mean = np.linalg.pinv(Laplacian(size))
    mean = np.array([[ 0.5, -0.5],[-0.5,  0.5]])*(2**(level_prime-2))
    init = tf.constant(np.random.normal(mean, sigma, size=(2,2)),dtype=tf.float32)
    #init = mean.astype(np.float32)
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
    c = tf.Variable(0.5*2**(level_prime),name='x{}'.format(level), trainable=True)
    D = c*D
    D = tf.expand_dims(tf.expand_dims(tf.expand_dims(D, 1), 0),0)
    Db = D * input_tensor
    return Db #this shape should be (1,1,8,1)

def output_layer(tensor):
    size = tf.shape(tensor)[0]
    return tensor - tf.reshape(tf.reduce_mean(tensor,2), [size,1,1,1])

def model(b, level, level_prime):
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
b = tf.placeholder("float", shape=[None,1,gridsize,1], name='b')
u_ = tf.placeholder("float", shape=[None,1, gridsize,1], name='u')

learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
output_u_ = model(b, level, level_prime)
output = tf.Print(output_u_, [output_u_])

loss = tf.losses.mean_squared_error(output_u_, u_)

hessian = tf.hessians(loss, tf.trainable_variables())
hessian = condense_hessian(hessian)


optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_placeholder, momentum=0.9, use_nesterov=True)
train = optimizer.minimize(loss)

correct = tf.equal(output_u_, u_)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



def network_training():
    init = tf.global_variables_initializer()
    #saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()
    sess.run(init)

    loss_plt = [0]*(epoch_num*num_batch)
    epoch = [0]*(epoch_num*num_batch)
    shift = 0
    for n in range(num_batch):
        b_vals, actual_u_outputs = mg_system_data.gen_data(gridsize, num_data, dim, equation)
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
            error = sess.run(loss, {b: np.array(b_vals), u_: np.array(actual_u_outputs), learning_rate_placeholder: learning_rate_prime})
            print('epoch # ', e+shift, 'training_error =', error)
            if math.isnan(error):
                break
            #print ('output_u_', sess.run(output_u_, {b: np.array(b_vals), u_: np.array(actual_u_outputs), learning_rate_placeholder: learning_rate}))
            loss_plt[e+shift] = error
            epoch[e+shift] = e+shift
            learning_rate_prime = learning_rate if (error_prime < error or error > 300) else (learning_rate*10)
            learning_rate_prime = learning_rate_prime if (error_prime < error or error > 50) else (learning_rate_prime*100)
            learning_rate_prime = learning_rate_prime if (error_prime < error or error > 3) else (learning_rate_prime*10)
            learning_rate_prime = learning_rate_prime if (error_prime < error or error > 0.05) else (learning_rate_prime*100)
            #learning_rate_prime = learning_rate_prime if (error_prime < error or error > 0.01) else (learning_rate_prime*10)
            error_prime = error
            
            hess = sess.run(hessian, {b: batch_x.astype(np.float32), u_: batch_y.astype(np.float32), learning_rate_placeholder: learning_rate_prime})
            hess_cond_num = cond_num(hess)
            print("hessian = %s" % hess_cond_num)
            
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
    plt.plot(epoch,loss_plt)
    plt.title("Loss Plot")
    plt.xlabel("Cumulative Epoch")
    plt.ylabel("Mean squared error")
    plt.yscale('log')
    #plt.show()
    return error,converged_vars, init_vars, epoch, loss_plt 


network_training()
