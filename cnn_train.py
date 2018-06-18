#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:59:32 2018

@author: sarahgage
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:49:33 2018

@author: sarahgage

This file is for playing around with different optimizers
"""

#Run the following for tensorboard tensorboard --logdir=/tmp/summary/  --host=127.0.0.1 --port=6006


import tensorflow as tf
import numpy as np
import math
import mg_system_data
import matplotlib.pyplot as plt
import hessian_comp
tf.reset_default_graph()
#sess = tf.InteractiveSession()
from tensorflow.python.ops.control_flow_ops import cond
import xlwt

#32 1.6
#64 0.8
#128 0.6
#256 0.6
#512 0.2
#1024 0.0 with low learning rate or overfits

#100 full lr 1e-8 0.6 good
#100 full 0.8 lr 1e-10 good; 10/100 better (hitting local min)
#100/1000 1.0 learning slowly large error but not inf

'''console 1/A is learning_rate = 1e-6
level = 5
level_prime = 0
batch_size = 100
num_data =100
gridsize = 64
epoch_num = 10000
num_batch = 1
dim = 1 #number of dimensions 
mean = np.reshape([0.5, 1, 0.5], [1,3,1,1])
sigma = 1.0'''

#with batch 16/256 and grid 128 and sigma=1, convergence 6/3
#grid 256, needed 1-e8 learning rate 



#Variables
learning_rate = 1e-8
level = 3
level_prime = 0
batch_size = 16
num_data = 256
gridsize = 16
epoch_num = 1
num_batch = 1
dim = 1 #number of dimensions 
mean = np.reshape([0.5, 1, 0.5], [1,3,1,1])
sigma = 1.0
equation = 'laplacian'

def Laplacian(n, stencil=[-1, 2, -1], periodic=True):
    A = stencil[1] * np.eye(n) + stencil[2] * np.eye(n, k=1) + stencil[0] * np.eye(n, k=-1)
    if periodic:
        A[0,-1] = stencil[0]
        A[-1,0] = stencil[2]
    return A

def cons(x):
    return tf.constant(x, dtype=tf.float32)

def replace_none_with_zero(l):
    
    #return [0 if i==None else i for i in l] 
    return tf.where(tf.is_nan(l), tf.zeros_like(l), l)

def vectorize(vars):
    vec = tf.convert_to_tensor([],dtype=tf.float32)
    for var in vars:
        var = tf.reshape(var, [tf.size(var)])
        vec = tf.concat([vec,var],0)
    #vec = replace_none_with_zero(vec) #NEED TO DO SOMETHING WITH THIS!
    return vec

def compute_hessian(fn, vars):
    '''vec = tf.convert_to_tensor([],dtype=tf.float32)
    for var in vars:
        size = tf.size(var)
        if size == None:
            print ('yes something is going on')
            continue
#            size = 1
        var = tf.reshape(var, [tf.size(var)])
        #vec.append(tf.reshape(var, [tf.size(var)]))
        vec = tf.concat([vec,var],0)
    #vec = replace_none_with_zero(vec) #NEED TO DO SOMETHING WITH THIS!
    #vec = tf.convert_to_tensor(vec)
    #hessian = tf.hessians(fn, vec)
    hessian = tf.gradients(tf.gradients(fn, vec), vec)
    
    
    #attempt 1
    #grads = tf.gradients(fn, vars)
    #hess0 = replace_none_with_zero(tf.gradients([grads[0]], vars))
    #hess1 = replace_none_with_zero(tf.gradients([grads[1]], vars))
    #hessian = tf.stack([tf.stack(hess0), tf.stack(hess1)])
    
    #attempt 2
    #grads = tf.gradients(fn, vars)
    #hess = replace_none_with_zero(tf.gradients(grads, vars))
    #hessian = hess
    
#    hessian = tf.hessians(fn, vars)
    
    """
    mat = []
    for v1 in vars:
        temp = []
        for v2 in vars:
            # computing derivative twice, first w.r.t v2 and then w.r.t v1
            temp.append(tf.gradients(tf.gradients(fn, v2)[0], v1)[0])
        temp = [cons(0) if t == None else t for t in temp] # tensorflow returns None when there is no gradient, so we replace None with 0
        temp = tf.stack(temp)
        mat.append(temp)
    mat = tf.stack(mat)"""'''
    hessian = []
    return hessian


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
    
def condense_hessian(h_vars):
    v = []
    for var in h_vars:
        v.append(tf.squeeze(var))
    return v

    
b = tf.placeholder("float", shape=[None,1,gridsize,1], name='b')
u_ = tf.placeholder("float", shape=[None,1, gridsize,1], name='u')

learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
output_u_ = model(b, level, level_prime)
output = tf.Print(output_u_, [output_u_])

loss = tf.losses.mean_squared_error(output_u_, u_)

hessian = tf.hessians(loss, tf.trainable_variables())
hessian = condense_hessian(hessian)
#vec_var = vectorize(tf.trainable_variables())

#loss = tf.losses.mean_pairwise_squared_error(output_u_, u_) 

#loss = (tf.norm(output_u_-u_))

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_placeholder, momentum=0.9, use_nesterov=True)
train = optimizer.minimize(loss)

correct = tf.equal(output_u_, u_)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



def network_training():
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()
    sess.run(init)
    #saver.save(sess,'my_test_model',global_step = 1000)
    #fw = tf.summary.FileWriter("/tmp/summary", sess.graph)

    loss_plt = [0]*(epoch_num*num_batch)
    epoch = [0]*(epoch_num*num_batch)
    shift = 0
    for n in range(num_batch):
        print ('n = {}'.format(n))
        b_vals, actual_u_outputs = mg_system_data.gen_data(gridsize, num_data, dim, equation)
        #print ('b_vals = {}'.format(b_vals))
        init_vars = {}
        i=1
        for var in tf.trainable_variables():
            var_val = sess.run(var)
            init_vars['{}'.format(var.name)]= var_val
            print(np.shape(var_val))
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
            #print ('output_u_', sess.run(output_u_, {b: np.array(b_vals), u_: np.array(actual_u_outputs), learning_rate_placeholder: learning_rate}))
            loss_plt[e+shift] = error
            epoch[e+shift] = e+shift
            learning_rate_prime = learning_rate if (error_prime < error or error > 300) else (learning_rate*10)
            learning_rate_prime = learning_rate_prime if (error_prime < error or error > 50) else (learning_rate_prime*100)
            learning_rate_prime = learning_rate_prime if (error_prime < error or error > 3) else (learning_rate_prime*10)
            learning_rate_prime = learning_rate_prime if (error_prime < error or error > 0.05) else (learning_rate_prime*100)
            #learning_rate_prime = learning_rate_prime if (error_prime < error or error > 0.01) else (learning_rate_prime*10)
            error_prime = error
            
            # WHAT TO DO WITH THE COMPUTED HESSIAN BELOW?
            #hess = sess.run(hessian, {b: batch_x.astype(np.float32), u_: batch_y.astype(np.float32), learning_rate_placeholder: learning_rate_prime})
            #xs = sess.run(vec_var)
            #print('vector of variables = {}'.format(xs))
            #ys = sess.run(loss, {b: batch_x.astype(np.float32), u_: batch_y.astype(np.float32), learning_rate_placeholder: learning_rate_prime}) 
            #print('ys = {}'.format(ys))
            #xs = tf.Variable(xs,name='xs')
            #init_a = tf.initialize_variables([xs])
            #sess.run(init_a)
            #grads = sess.run(tf.gradients(loss, xs), {b: batch_x.astype(np.float32), u_: batch_y.astype(np.float32), learning_rate_placeholder: learning_rate_prime})
            #print("gradients = {}".format(grads))
            #print ('trainable variable = {}'.format(sess.run(tf.trainable_variables()[0:3])))
            #test_stack = tf.stack([tf.reshape(tf.trainable_variables()[0],[1,1,1,1]),tf.trainable_variables()[1]],0)
            #print ('test_stack = {}'.format(sess.run(test_stack)))
            hess = sess.run(hessian, {b: batch_x.astype(np.float32), u_: batch_y.astype(np.float32), learning_rate_placeholder: learning_rate_prime})
            #hess = sess.run(hessian)
            #hess = sess.run(tf.gradients(tf.convert_to_tensor(ys),vec_var))
            #hess = sess.run(tf.squeeze(tf.hessians(loss, tf.trainable_variables()[1])), {b: batch_x.astype(np.float32), u_: batch_y.astype(np.float32), learning_rate_placeholder: learning_rate_prime})
            print("hessian = %s" % hess)
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
        converged_vars = {}
        #converged_u = sess.run(output_u_, {b: np.array(b_vals), u_: np.array(actual_u_outputs)})
        j=i+1
        for var in tf.trainable_variables():
            var_val = sess.run(var)
            #sh.write(trial+1,j,'{}'.format(var_val))
            #print ('var name = ', var.name,' values = ', var_val)
            converged_vars['{}'.format(var.name)]= var_val
            j+=1
        print ('initial params = {}'.format(init_vars))
        print ('converged params = {}'.format(converged_vars))
        #print('epoch # ', e+shift, 'training_error =', error)
    plt.plot(epoch,loss_plt)
    plt.title("Loss Plot")
    plt.xlabel("Cumulative Epoch")
    plt.ylabel("Mean squared error")
    plt.yscale('log')
    #plt.show()
    return error,converged_vars, init_vars, epoch, loss_plt 


network_training()
