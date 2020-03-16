###############################################################################################################
# NB: all the sentences after # are comments

# This is the code that I have used for our simulations


# HERE WE ARE IMPORTING ONLY SOME LIBRARIES
import numpy as np  # this is a good math library in python, released by Berkley University
import tensorflow as tf  # importing TensorFlow
from scipy.stats import multivariate_normal as normal  # math library for calling a normal distribution
import time  # library for fix the time
import sys  # library for some default functions

## Here the code imports some specifics from tensorflow library

from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.training.moving_averages import assign_moving_average


#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################

# NEURAL NETWORK DEFINITION: START#

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################


# Here we are defining the subnetwork
# The function takes as INPUT a tensor x of dimension batch_size * d
# and retunrs as OUTPUT a tensor output of size batch_size * d.
# The function normalises the tensor x using batch normalisation
# and then computes an iterative loop for obtaining the tensor output.
# The Neural Network works in the following way:
# 1) The INPUT layer is normalised using batch normalisation, the layer is composed by d neurons
# 2) The second layer is fed with the OUTPUT of step 1.
# 3) The third layer is fed with the OUTPUT of step 2.
# 4) The fourth layer is fed with the OUTPUT of step 3.
def _subnetwork(x, neu__, name):
    with tf.variable_scope(name):
        # this is the first layer, i.e. the INPUT layer
        hiddens = _batch_norm(x, name='path_input_norm')  # I give to function batch norm tensor x

        for i in range(1, len(neu__) - 1):
            hiddens = _dense_batch_layer(hiddens, neu__[i], activation_fn=tf.nn.relu, name='layer_{}'.format(i))
            # here the second and third layer are computed. The activation function is given to the layer

        output = _dense_batch_layer(hiddens,    neu__[-1], activation_fn=None, name='final_layer')
        # here the last layer is computed
    return output

# The following function describes the single layer.
# As you can see this function takes as INPUT a tensor input_, the size of the layer, i.e. the number of neurons, and the activation function.
def _dense_batch_layer(input_, output_size, activation_fn=None, stddev=5.0, name='linear'):
    with tf.variable_scope(name):
        shape = input_.get_shape().as_list()
        weight = tf.get_variable('Matrix', [shape[1], output_size], tf.float64,
                                 tf.random_normal_initializer(stddev=stddev / np.sqrt(shape[1] + output_size)))
        hiddens = tf.matmul(input_,
                            weight)  # This is the multiplication between the matrix of weights and the vector of input
        hiddens_bn = _batch_norm(hiddens, is_training)  # Again is normalised the result
    if activation_fn:
        # the activation function, i.e. the relu, is applied only on the second and third layer, as described in Implementation section of "Solving high dimensional partial differential equations using deep learning"
        return activation_fn(hiddens_bn)
    else:
        return hiddens_bn


# The following function defines the batch normalisation
# Again as INPUT is given a tensor x, other stuff are not important for the conceptual understanding of neural network.
# This normalisation is really important for numerical studies using Deep Learning.
# Indeed, training Deep Neural Network is complicated by the fact that the distribution of each Layer's input changes during training
# as the parameters of the previous layers change. This slow down the training by requiring lower learning rates and carful
# parameter initialisation. This methods, instead, normalising each batch, allows to using higher learning rates and allows to
# speed up the algorithm.

# Batch normalisation normalises each batch independetnly for having mean 0 and variance 1. For a layer with d dimensional input, for
# example, batch normalisation will normalises each dimension, where the expectation and the variance are computed over the training data.
# This normalisation speeds up the convergence, even when the features are correleted.
# It is important to note that normlising each input of a layer may change what the layer can represent. To avoid this, batch normalisation
# makes sure that the transformation inserted in the network can represent the identity transformation. To accomplish this, it introduces, for each activation input, i.e. for each dimension,
# a pair of parameters gamma and beta, which scale and shift the nomalised value.
# These parameters are learned along with the original model parameters, and restore the representation power of the network.
def _batch_norm(x, affine=True, name='batch_norm'):
    """Batch normalization"""
    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]
        # beta parameter. It needs to be learned
        beta = tf.get_variable('beta', params_shape, tf.float64,
                               initializer=tf.random_normal_initializer(
                                   0.0, stddev=0.1, dtype=tf.float64))
        # gamma parameter. It needs to be learned
        gamma = tf.get_variable('gamma', params_shape, tf.float64,
                                initializer=tf.random_uniform_initializer(
                                    0.1, 0.5, dtype=tf.float64))
        moving_mean = tf.get_variable('moving_mean', params_shape, tf.float64,
                                      initializer=tf.constant_initializer(0.0, tf.float64),
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance', params_shape, tf.float64,
                                          initializer=tf.constant_initializer(1.0, tf.float64),
                                          trainable=False)
        # These ops will only be preformed when training
        mean, variance = tf.nn.moments(x, [0], name='moments')
        _extra_train_ops.append(assign_moving_average(moving_mean, mean, 0.99, True))
        _extra_train_ops.append(assign_moving_average(moving_variance, variance, 0.99, False))
        mean, variance = tf.cond(is_training,
                                 lambda: (mean, variance),
                                 lambda: (moving_mean, moving_variance))
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-6)
        y.set_shape(x.get_shape())
        return y


#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################

# NEURAL NETWORK DEFINITION: END#

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################


#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################

# DEFINITION FUNCTIONS FOR INTEGRATION: START#

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################

# It is the implementation of equation 9 and 10 in "Solving high dimensional partial differential equations using deep learning"
def _f_t(y):
    return 0.


# This function returns trajectories and white noises
def integrate(num_sample):
    _x_init = np.ones(d) * 0.0
    dw_sample = normal.rvs(size=[num_sample, d, N]) * np.sqrt(2.*dt)
    x_sample = np.zeros([num_sample, d, N + 1])
    x_sample[:, :, 0] = np.ones([num_sample, d]) * _x_init
    for i in range(0, N):
        x_sample[:, :, i  + 1] = x_sample[:, :, i]  + ( dw_sample[:, :, i])
    return dw_sample, x_sample

# This function returns trajectories and white noises
def integrate_init(num_sample, _rn_numb_init):
    dw_sample = normal.rvs(size=[num_sample, d, N]) * np.sqrt(2.*dt)
    x_sample = np.zeros([num_sample, d, N + 1])
    x_sample[:, :, 0] = _rn_numb_init
    for i in range(0, N):
        x_sample[:, :, i + 1] = x_sample[:, :, i]  + ( dw_sample[:, :, i])
    return dw_sample, x_sample

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################

# DEFINITION FUNCTIONS FOR INTEGRATION: END#

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################


#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################

# FROM HERE WE START#

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################



#exact solution
#x is a d dimensional veector
def exact_sol(x):
    A=float(T)*float(d)
    B=np.sum(0.5*(x*x))
    return A+B

def rel_err(x,y):
    return np.abs((x-y)/x)


# Defintion terminal conditions in tensor flow
def phi(x):
    B=tf.reduce_sum(0.5*(x*x), 1, keepdims=True)
    return B


# This equation prints on file the values that we need to know
def _approximate():
    x=np.zeros([d])
    s=exact_sol(x)
    lr, gs, _loss, _u_0 = sess.run([learning_rate, global_step, loss, u_0], feed_dict=feed_dict_valid)
    t1_train = time.time()
    file_out.write('% i\t '' % f \t % f \t % f \t %f \t %f \n' % (gs, lr, t1_train - t0_train, _loss, _u_0[0], s))
    file_out.flush()  # flush files


tf.compat.v1.reset_default_graph()  # default function in TensorFlow that needs to be used only at the beginning, never in Session() or in run()
dtype = tf.float64  # Defining the type of the tensor.

with tf.compat.v1.Session() as sess:  # TensorFlow's session starts
    sample = int(sys.argv[1])
    file_name = 'Example_' + str(sample) + '.txt'
    MC=10000
    T, N = 1., 40 # T is the final time, N is the number of the step we make for each trajectory
    dt = T / N # dt is the step interval that we need to use in the integration
    batch_size = int(sys.argv[2]) # this is the size for each batch
    interval_point= float(sys.argv[3])/2.
    d = int(sys.argv[4]) #d is the number of dimensions
    min = 0. - interval_point
    max = 0. + interval_point
    file_name_approx_values_single_experiment = 'approx_values_single_experiment_' + str(sample) + '_batch_size_' + str(batch_size)  + '_d_' + str(d) + '.txt'
    file_name_rel_error = 'rel_error_batch_size_' + str(batch_size) + '_d_' + str(d) + '.txt'
    rn_numb_init = np.random.uniform(min, max,size=[batch_size, d]) # this is the tensor of random initial points where trajectories start
    neurons = [d, d + 10, d + 10, d]  # this array tells how many neurons needs to have each layer, i.e. the first layer has d neurons, the second one d+10, the third one d+10 and the fourth one d
    _neurons_hessian = [d, d + 10, d + 10, 1]
    neurons_u0 = [d, d + 10, d + 10, 1]
    train_steps = 15000  # this is the number of training steps
    mc_freq = 100  # global variable used for printing
    lr_boundaries = [500, 1500, 2500, 5500, 10000]  # defines when we want to change the learning value
    #lr_boundaries = [4000, 6000]  # defines when we want to change the learning value
    #lr_values = [0.005,0.001, 0.0005]  # learning values
    lr_values = [10., 5., 1., 0.1, 0.001, 0.0001]  # learning values
    _extra_train_ops = []  # array for storing variables that needs to be optimised

    t0_train = time.time()  # we collect the starting time

    # we are defining some instruction for TensorFlow.
    DX = tf.compat.v1.placeholder(tf.float64, [None, d, N + 1 ], name='X')
    DW = tf.compat.v1.placeholder(tf.float64, [None, d, N], name='dW')
    is_training = tf.compat.v1.placeholder(tf.bool)

    with tf.variable_scope("forward"):
        # Definition of u_0.
        u_0 = _subnetwork(DX[:, :, 0], neurons_u0, str(0)) / d
        # definition of variable of u_0.
        grad_u_0 = _subnetwork(DX[:, :, 0], neurons, str(-1)) / d
        u_approx = u_0
        grad_u_t = grad_u_0
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    # INTEGRATION FOR u_t*
    
        for t in range(0, N - 1):
            
            g_s_t = str(t + 1)
            u_approx = u_approx - dt * _f_t(u_approx) + tf.reduce_sum(grad_u_t * DW[:, :, t], 1, keepdims=True)
            grad_u_approx_t = _subnetwork(DX[:, :, t + 1], neurons, g_s_t) / d
            grad_u_t = grad_u_approx_t
        
        
        u_approx = u_approx - dt * _f_t(u_approx) + tf.reduce_sum(grad_u_t * DW[:, :, -1], 1, keepdims=True) 
   
    # u_approx stores the last value of the integration, i.e. the one that we need for the loss function
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    # END INTEGRATION FOR u_t

    phi_final = tf.convert_to_tensor(DX[:, :, -1], dtype=dtype)
    delta = u_approx - phi(phi_final)
    DELTA_CLIP = 50.0
    # definition of loss function, more precisely this function makes the following instructions:
    # if delta is less then 50 then the function computes the reduced mean square of delta
    # else the function does a trigger and computes 100 * |delta| - 50^2 for not having big numbers as output file
    loss = tf.reduce_mean(
        tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta), 2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))

    # definition of variables for optimisation using TensorFlow
    global_step = tf.compat.v1.get_variable('global_step', [], tf.int32, tf.constant_initializer(0), trainable=False)
    learning_rate = tf.compat.v1.train.piecewise_constant(global_step, lr_boundaries, lr_values)

    # from here we use the sintax of tensor flow for minimising the loss function and optimising the weights of the neural networks
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(loss, trainable_variables)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    apply_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step, name='train_step')
    all_ops = [apply_op] + _extra_train_ops
    train_op = tf.group(*all_ops)

    # open file for writing
    file_out = open(file_name, 'w')
    file_out_approx_values = open(file_name_approx_values_single_experiment, 'w')
    file_out_rel_error = open(file_name_rel_error, 'a')
    # test set, i.e. we create 256 trajectories
    dw_valid, x_valid = integrate(2)
    feed_dict_valid = {DW: dw_valid, DX: x_valid, is_training: False}  # definition of a command for optimisation
    # random initialisation weights and variables
    sess.run(tf.compat.v1.global_variables_initializer())
    # loop for training our variables
    for step in range(train_steps + 1):

        if step % mc_freq == 0:
            print(step)
            # if the step is a multiple of mc_freq then print on file
            _approximate()
        # definition of training set of size equal to batch size.
        # these trajectories are computed at each step
        dw_train, x_train = integrate_init(batch_size, rn_numb_init)
        # command for training and optimise everything.
        sess.run(train_op, feed_dict={DX: x_train, DW: dw_train, is_training: True})
        # end for
    # we print last value on file
    _approximate()


    rn_numb_test = np.random.uniform(min, max,size=[MC, d]) # this is the tensor of random test points
    dw_test, x_test = integrate_init(MC, rn_numb_test)
    test_u_0 = sess.run([u_0], feed_dict={DX: x_test, DW: dw_test, is_training: False})
    # I prepare the array of approx value of u given by NN
    a_test=np.array(test_u_0)
    rel_err_arr=np.zeros([MC])

    for i in range(0, MC):
        x=rn_numb_test[i, :]
        y=a_test[0,i]
        res=np.array(exact_sol(x))
        re = np.array(rel_err(res, y))
        rel_err_arr[i]=re
        file_out_approx_values.write('%f \t %f \t %f \n' % (res, y, rel_err_arr[i]))
        file_out_approx_values.flush() # flush files


    sum_rel_error = np.sum(rel_err_arr)
    mean_rel_error = np.mean(rel_err_arr)
    std_rel_error = np.std(rel_err_arr)
    file_out_rel_error.write('%i \t '' %f \t %f \t %f \t %f \n' % (batch_size, sum_rel_error, mean_rel_error, std_rel_error,train_steps))
    file_out_rel_error.flush() # flush files



    file_out.close()
    file_out_approx_values.close()
    file_out_rel_error.close()

