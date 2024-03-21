import numpy as np
import tensorflow as tf
import time

VARIABLE_COUNTER = 0


NUM_CH = [64,128,256,512,1024]
KER_SZ = 3

########## TABLE 1 ARCHITECTURE PARAMETERS #########
color_encode = "ABCDEFG"
block_in  = [128,128,128,128,256,256,256]
block_out = [64,128,128,256,256,256,128]
block_inter = [64,32,64,32,32,64,32]
block_conv1 = [1,1,1,1,1,1,1]
block_conv2 = [3,3,3,3,3,3,3]
block_conv3 = [7,5,7,5,5,7,5]
block_conv4 = [11,7,11,7,7,11,7]

####################### TABLE 1 END#################

def variable(name, shape, initializer,regularizer=None):
	global VARIABLE_COUNTER
	with tf.device('/cpu:0'):
		VARIABLE_COUNTER += np.prod(np.array(shape))
		return tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=tf.float32, trainable=True)


def conv_layer(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1,bn=False,training=False,relu=True):
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		conv_layer = tf.nn.bias_add(conv, biases)
		if bn:
			conv_layer = batch_norm_layer(conv_layer,scope,training)
		if relu:
			conv_layer = tf.nn.relu(conv_layer, name=scope.name)
	return conv_layer
	
def conv_layer_sigmoid(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1,bn=False,training=False,sigmoid=True):
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		conv_layer = tf.nn.bias_add(conv, biases)
		if bn:
			conv_layer = batch_norm_layer(conv_layer,scope,training)
		if sigmoid:
			conv_layer = tf.nn.sigmoid(conv_layer, name=scope.name)			
# 	print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),conv_layer.get_shape().as_list()))
	return conv_layer

def conv_layer_nobias(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1,bn=False,training=False,relu=True):
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		# biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		# conv_layer = tf.nn.bias_add(conv, biases)
		conv_layer = conv
		if bn:
			conv_layer = batch_norm_layer(conv_layer,scope,training)
		if relu:
			conv_layer = tf.nn.relu(conv_layer, name=scope.name)			
# 	print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),conv_layer.get_shape().as_list()))
	return conv_layer

def conv_layer_tanh(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1,bn=False,training=False,tanh=True):
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		conv_layer = tf.nn.bias_add(conv, biases)
		if bn:
			conv_layer = batch_norm_layer(conv_layer,scope,training)
		if tanh:
			conv_layer = tf.nn.tanh(conv_layer, name=scope.name)			
# 	print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),conv_layer.get_shape().as_list()))
	return conv_layer

def max_pooling(input_tensor,name,factor=2):
	pool = tf.nn.max_pool(input_tensor, ksize=[1, factor, factor, 1], strides=[1, factor, factor, 1], padding='SAME', name=name)
# 	print('Pooling layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),pool.get_shape().as_list()))
	return pool


def batch_norm_layer(input_tensor,scope,training):
	return tf.contrib.layers.batch_norm(input_tensor,scope=scope,is_training=training,decay=0.99)

def batch_norm_layer1(input_tensor,scope):
	return tf.contrib.layers.batch_norm(input_tensor,scope=scope,decay=0.99)
def R_3(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1):
	input_channels = input_tensor.get_shape().as_list()[-1]
	# print('input_channels')
	# print(input_channels)
	with tf.variable_scope(name) as scope:
		# kernel_size=5
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		x0=tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		x0=batch_norm_layer1(x0,scope)
		x0=tf.nn.relu(x0,name=scope.name)
		# print('x0',x0.shape) #(1, 256, 256, 64)
		kernel1 = variable('weights1', [kernel_size, kernel_size, 64, 64], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		x1=tf.nn.conv2d(x0, kernel1, [1, stride, stride, 1], padding='SAME')
		x1=batch_norm_layer1(x1,scope)
		x1=tf.nn.relu(x1,name=scope.name)
		# print('x1',x1.shape) #(1, 256, 256, 64)
		kernel2 = variable('weights2', [kernel_size, kernel_size, 64, 1], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		pre_depth=tf.nn.conv2d(x1, kernel2, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [1], tf.constant_initializer(0.0))
		pre_depth = tf.nn.bias_add(pre_depth, biases)
		# print('pre_depth',pre_depth.shape) #(1, 256, 256, 1)
		kernel3 = variable('weights3', [kernel_size, kernel_size, 64, 8], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		h=tf.nn.conv2d(x1, kernel3, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases1', [8], tf.constant_initializer(0.0))
		h = tf.nn.bias_add(h, biases)
	
	return h,pre_depth
  
def F_t(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1): 
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:	
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		convf = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		convf = tf.nn.bias_add(convf, biases)
		convf = tf.nn.sigmoid(convf,name=scope.name)
	return convf
	
def I_t(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1): 
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		convi = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		convi = tf.nn.bias_add(convi, biases)
		convi = tf.nn.sigmoid(convi,name=scope.name)
	return convi
		
def C_t(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1): 
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		convc = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		convc = tf.nn.bias_add(convc, biases)
		convc = tf.nn.tanh(convc,name=scope.name)
	return convc
		
def Q_t(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1): 
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		convq = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		convq = tf.nn.bias_add(convq, biases)
		convq = tf.nn.sigmoid(convq,name=scope.name)
	return convq

def hourglass_refinement(netIN_1,netIN_2,netIN_3,netIN_4,netIN_5,training):
	print('-'*30)
	print('Modify_Clstm_order Hourglass Architecture')
	print('-'*30)
	global VARIABLE_COUNTER
	VARIABLE_COUNTER = 0;
	layer_name_dict = {}
	def layer_name(base_name):
		if base_name not in layer_name_dict:
			layer_name_dict[base_name] = 0
		layer_name_dict[base_name] += 1
		name = base_name + str(layer_name_dict[base_name])
		return name


	bn = True
	def hourglass_stack_fused_depth_prediction(stack_in):
		c0 = conv_layer(stack_in,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		c1 = conv_layer(c0,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		c2 = conv_layer(c1,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)

		p0 = max_pooling(c2,layer_name('pool'))
		c3 = conv_layer(p0,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)

		p1 = max_pooling(c3,layer_name('pool'))
		c4 = conv_layer(p1,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)

		p2 = max_pooling(c4,layer_name('pool'))
		c5 = conv_layer(p2,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)

		p3 = max_pooling(c5,layer_name('pool'))
		c6 = conv_layer(p3,layer_name('conv'),KER_SZ,NUM_CH[4],bn=bn,training=training)

		c7 = conv_layer(c6,layer_name('conv'),KER_SZ,NUM_CH[4],bn=bn,training=training)
		c8 = conv_layer(c7,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)

		r0 = tf.image.resize_images(c8,[c8.get_shape().as_list()[1]*2, c8.get_shape().as_list()[2]*2])
		cat0 = tf.concat([r0,c5],3)

		c9 = conv_layer(cat0,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)
		c10 = conv_layer(c9,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)

		r1 = tf.image.resize_images(c10,[c10.get_shape().as_list()[1]*2, c10.get_shape().as_list()[2]*2])
		cat1 = tf.concat([r1,c4],3)

		c11 = conv_layer(cat1,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)
		c12 = conv_layer(c11,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)

		r2 = tf.image.resize_images(c12,[c12.get_shape().as_list()[1]*2, c12.get_shape().as_list()[2]*2])
		cat2 = tf.concat([r2,c3],3)

		c13 = conv_layer(cat2,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)
		c14 = conv_layer(c13,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)

		r3 = tf.image.resize_images(c14,[c14.get_shape().as_list()[1]*2, c14.get_shape().as_list()[2]*2])
		cat3 = tf.concat([r3,c2],3)

		c15 = conv_layer(cat3,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		c16 = conv_layer(c15,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)# c16: 10*64*256*256
		# stack_out_d = conv_layer(c16, layer_name('conv'), 1, 1, bn=False, training=training, relu=False)
		# return stack_out_d
		return c16

	
	
	def R_CLSTM(f_d,h_state,c_state):
		input_cat1 = tf.concat([f_d, h_state], 3)
		q_output_channels = f_d.get_shape().as_list()[-1]
		F_t1=F_t(input_cat1,layer_name('r_clstm'),KER_SZ,8)   
		I_t1=I_t(input_cat1,layer_name('r_clstm'),KER_SZ,8)   
		C_t1=C_t(input_cat1,layer_name('r_clstm'),KER_SZ,8)      
		Q_t1=Q_t(input_cat1,layer_name('r_clstm'),KER_SZ,q_output_channels)   
		c_state1 = F_t1 * c_state + I_t1 * C_t1 
		
		# print('c_state1, Q_t1')
		# print(F_t1.shape,I_t1.shape,c_state1.shape,Q_t1.shape,c_state1.shape) # (1, 256, 256, 8) (1, 256, 256, 8) (1, 256, 256, 8) (1, 256, 256, 64) (1, 256, 256, 8)
		# print(3/0)
        
		input_cat2=tf.concat([c_state1, Q_t1], 3)
		# print(input_cat2.shape) #(1, 256, 256, 72)

		h_state1, p_depth1 = R_3(input_cat2,layer_name('r_clstm'),KER_SZ,q_output_channels) 
		# print('h_state1, p_depth1')
		# print(input_cat2.shape,h_state1.shape,p_depth1.shape) #(1, 16, 16, 512) (1, 16, 16, 1)
		# print(3/0)
		return h_state1,c_state1,p_depth1
	
	def R_CLSTM_5(f_d1,f_d2,f_d3,f_d4,f_d5): #f_d.shape=[10,256,256,64]
		# input_tensor = maps_2_cubes(f_d, b, d) #[2,5,256,256,64],bsize=2
		d,h, w,c = f_d1.get_shape().as_list() 
		print('net:',d,h,w,c) 

		h_state_init = tf.zeros([1, h, w, 8])
		c_state_init = tf.zeros([1, h, w, 8])
		# seq_len = 5
		h_state, c_state = h_state_init, c_state_init
		# output_inner = variable('out_10', [2, 5,h, w, 8], tf.constant_initializer(0.0))
		# output_inner = tf.zeros([2, 5,h, w, 8])
		h_state1,c_state1,p_depth1=R_CLSTM(f_d1,h_state,c_state)
		h_state2,c_state2,p_depth2=R_CLSTM(f_d2,h_state1,c_state1)
		h_state3,c_state3,p_depth3=R_CLSTM(f_d3,h_state2,c_state2)
		h_state4,c_state4,p_depth4=R_CLSTM(f_d4,h_state3,c_state3)
		h_state5,c_state5,p_depth5=R_CLSTM(f_d5,h_state4,c_state4)

		# input_cat1 = tf.concat([f_d1, h_state], 3)
		# F_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		# I_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		# C_t1=conv_layer_tanh(input_cat1,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		# Q_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		# c_state1 = F_t1 * c_state + I_t1 * C_t1 #c_state.shape [2,256,256,8]
		# h_state1, p_depth1 = R_3(tf.concat([c_state1, Q_t1], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]
        
		# input_cat2 = tf.concat([f_d2, h_state1], 3)
		# F_t2=conv_layer_sigmoid(input_cat2,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		# I_t2=conv_layer_sigmoid(input_cat2,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		# C_t2=conv_layer_tanh(input_cat2,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		# Q_t2=conv_layer_sigmoid(input_cat2,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		# c_state2 = F_t2 * c_state1 + I_t2 * C_t2 #c_state.shape [2,256,256,8]
		# h_state2, p_depth2 = R_3(tf.concat([c_state2, Q_t2], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]
        
		# input_cat3 = tf.concat([f_d3, h_state2], 3)
		# F_t3=conv_layer_sigmoid(input_cat3,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		# I_t3=conv_layer_sigmoid(input_cat3,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		# C_t3=conv_layer_tanh(input_cat3,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		# Q_t3=conv_layer_sigmoid(input_cat3,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		# c_state3 = F_t3 * c_state2 + I_t3 * C_t3 #c_state.shape [2,256,256,8]
		# h_state3, p_depth3 = R_3(tf.concat([c_state3, Q_t3], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]
        
		# input_cat4 = tf.concat([f_d4, h_state3], 3)
		# F_t4=conv_layer_sigmoid(input_cat4,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		# I_t4=conv_layer_sigmoid(input_cat4,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		# C_t4=conv_layer_tanh(input_cat4,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		# Q_t4=conv_layer_sigmoid(input_cat4,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		# c_state4 = F_t4 * c_state3 + I_t4 * C_t4 #c_state.shape [2,256,256,8]
		# h_state4, p_depth4 = R_3(tf.concat([c_state4, Q_t4], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]
        
		# input_cat5 = tf.concat([f_d5, h_state4], 3)
		# F_t5=conv_layer_sigmoid(input_cat5,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		# I_t5=conv_layer_sigmoid(input_cat5,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		# C_t5=conv_layer_tanh(input_cat5,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		# Q_t5=conv_layer_sigmoid(input_cat5,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		# c_state5 = F_t5 * c_state4 + I_t5 * C_t5 #c_state.shape [2,256,256,8]
		# h_state5, p_depth5 = R_3(tf.concat([c_state5, Q_t5], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]

		return p_depth1,p_depth2,p_depth3,p_depth4,p_depth5
	
	f_d1 = hourglass_stack_fused_depth_prediction(netIN_1)
	f_d2 = hourglass_stack_fused_depth_prediction(netIN_2)
	f_d3 = hourglass_stack_fused_depth_prediction(netIN_3)
	f_d4 = hourglass_stack_fused_depth_prediction(netIN_4)
	f_d5 = hourglass_stack_fused_depth_prediction(netIN_5)
    
	out1,out2,out3,out4,out5=R_CLSTM_5(f_d1,f_d2,f_d3,f_d4,f_d5)

	return out1,out2,out3,out4,out5
