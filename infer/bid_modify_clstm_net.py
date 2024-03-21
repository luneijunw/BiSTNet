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

def conv_layer_softmax(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1,bn=False,training=False,tanh=True):
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		conv_layer = tf.nn.bias_add(conv, biases)
		if bn:
			conv_layer = batch_norm_layer(conv_layer,scope,training)
		if tanh:
			conv_layer = tf.nn.softmax(conv_layer, dim=-1,name=scope.name)	
			# print('conv_layer.shape')	
			# print(conv_layer.shape)	 #(1, 16, 16, 2)
	return conv_layer

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
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, 768], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		x0=tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		x0=batch_norm_layer1(x0,scope)
		x0=tf.nn.relu(x0,name=scope.name)
		input_channels = x0.get_shape().as_list()[-1]
		# print('x0',x0.shape) #(1, 256, 256, 64)
		kernel1 = variable('weights1', [kernel_size, kernel_size, input_channels, 768], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		x1=tf.nn.conv2d(x0, kernel1, [1, stride, stride, 1], padding='SAME')
		x1=batch_norm_layer1(x1,scope)
		x1=tf.nn.relu(x1,name=scope.name)
		input_channels = x1.get_shape().as_list()[-1]
		# print('x1',x1.shape) #(1, 256, 256, 64)
		kernel2 = variable('weights2', [1, 1, input_channels, 1], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		pre_depth=tf.nn.conv2d(x1, kernel2, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [1], tf.constant_initializer(0.0))
		pre_depth = tf.nn.bias_add(pre_depth, biases)
		# print('pre_depth',pre_depth.shape) #(1, 256, 256, 1)
		kernel3 = variable('weights3', [kernel_size, kernel_size, input_channels, 512], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		h=tf.nn.conv2d(x1, kernel3, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases1', [512], tf.constant_initializer(0.0))
		h = tf.nn.bias_add(h, biases)
		# print('h',h.shape) #(1, 256, 256, 8)
	
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
	print('Modify_Clstm_Attention_fusion Hourglass Architecture')
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
		# print('0',c2.shape)
		p0 = max_pooling(c2,layer_name('pool'))
		# print('1',p0.shape)
		c3 = conv_layer(p0,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)
		# print('c3',c3.shape)
		p1 = max_pooling(c3,layer_name('pool'))
		# print('p1',p1.shape)
		c4 = conv_layer(p1,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)
		# print('c4',c4.shape)
		p2 = max_pooling(c4,layer_name('pool'))
		
		c5 = conv_layer(p2,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)
		
		p3 = max_pooling(c5,layer_name('pool'))
		
		c6 = conv_layer(p3,layer_name('conv'),KER_SZ,NUM_CH[4],bn=bn,training=training)  
		
		c7 = conv_layer(c6,layer_name('conv'),KER_SZ,NUM_CH[4],bn=bn,training=training)
		
		# print(3/0)
		return c2,c3,c4,c5,c7
	

	def attention(feature_forward,feature_backward):
		# feature_forward,feature_backward都是[1,256,256,512]
		feature_forward1=conv_layer_tanh(feature_forward,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training) 
		feature_backward1=conv_layer_tanh(feature_backward,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training) 
		f = tf.concat([feature_forward1,feature_forward1],3) #(1, 16, 16, 512)
		# print(f.shape) ##(1, 16, 16, 512)
		f=tf.reshape(f, [1,-1]) 
		# print(f.shape) ##(1, 16, 16, 131072)
		fc1=tf.keras.layers.Dense(units=256,input_shape=(256*2,))
		y1=fc1(f)
		y1=tf.nn.tanh(y1)
		fc2=tf.keras.layers.Dense(units=256,input_shape=(256,))
		y2=fc2(y1)
		y2=tf.nn.tanh(y2)
		fc3=tf.keras.layers.Dense(units=2,input_shape=(256,))
		y3=fc3(y2)
		y3=tf.nn.tanh(y3)
		a3=tf.nn.softmax(y3,dim=-1)
		# print('a3.shape')
		# print(a3.shape) #(1, 2)
		F1 = a3[0,0] * feature_forward #[1,16,16,512]
		F2 = a3[0,1] * feature_backward #[1,16,16,512]
		# print('F1.shape,F2.shape')
		# print(F1.shape,F2.shape)
		F3 = tf.concat([F1,F2],3) 
		# print('F3.shape')
		# print(F3.shape) #(1, 16, 16, 1024)
		# print(3/0)
		# F = conv_layer_tanh(F3,layer_name('conv'),KER_SZ,NUM_CH[4],bn=bn,training=training)
		# print('--------------------name---------------')
		# print('fc1',fc1.name,'fc2',fc2.name,'fc3',fc3.name,'y3',y3.name,'a3',a3.name,'F1',F1.name,'F2',F2.name,'F3',F3.name)	
		# print('F',F.name)	
		
        # print(3/0)        
		return a3,F3
	
	def fusion(c11,c12,c13,c14,F):
		c1 = conv_layer(F,layer_name('conv'),KER_SZ,NUM_CH[4],bn=bn,training=training)
		c2 = conv_layer(c1,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)
		r0 = tf.image.resize_images(c2,[c2.get_shape().as_list()[1]*2, c2.get_shape().as_list()[2]*2])
		cat0 = tf.concat([r0,c14],3)
		c3 = conv_layer(cat0,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)
		c4= conv_layer(c3,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)
		r1 = tf.image.resize_images(c4,[c4.get_shape().as_list()[1]*2, c4.get_shape().as_list()[2]*2])
		cat1 = tf.concat([r1,c13],3)
		c5 = conv_layer(cat1,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)
		c6 = conv_layer(c5,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)
		r2 = tf.image.resize_images(c6,[c6.get_shape().as_list()[1]*2, c6.get_shape().as_list()[2]*2])
		cat2 = tf.concat([r2,c12],3)
		c7= conv_layer(cat2,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)
		c8 = conv_layer(c7,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		r3 = tf.image.resize_images(c8,[c8.get_shape().as_list()[1]*2, c8.get_shape().as_list()[2]*2])
		cat3 = tf.concat([r3,c11],3)
		c15 = conv_layer(cat3,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		c16 = conv_layer(c15,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training) # c16: 1*256*256*64
		stack_out_d = conv_layer(c16, layer_name('conv'), 1, 1, bn=False, training=training, relu=False) #1,256,256,1
		# print('c16.shape,stack_out_d.shape')
		# print(c16.shape,stack_out_d.shape)
		# print(3/0)
		return stack_out_d
  
	# def R_3(input_cat):
	# 	x0=conv_layer_nobias(input_cat,layer_name('conv'),3,768,bn=bn,training=training)
	# 	x1=conv_layer_nobias(x0,layer_name('conv'),3,768,bn=bn,training=training)
	# 	h=conv_layer(x1,layer_name('conv'),KER_SZ,512,bn=False,relu=False)
	# 	pred_depth=conv_layer(x1,layer_name('conv'),1,1,bn=False,relu=False)
		# return h, pred_depth
	
	def R_CLSTM(f_d,h_state,c_state):
		input_cat1 = tf.concat([f_d, h_state], 3)
		q_output_channels = f_d.get_shape().as_list()[-1]
		F_t1=F_t(input_cat1,layer_name('r_clstm'),KER_SZ,256)   
		I_t1=I_t(input_cat1,layer_name('r_clstm'),KER_SZ,256)   
		C_t1=C_t(input_cat1,layer_name('r_clstm'),KER_SZ,256)      
		Q_t1=Q_t(input_cat1,layer_name('r_clstm'),KER_SZ,512)   
		c_state1 = F_t1 * c_state + I_t1 * C_t1
		input_cat2=tf.concat([c_state1, Q_t1], 3)
		# print('c_state1, Q_t1')
		# print(c_state1.shape, Q_t1.shape) #(1, 16, 16, 256) (1, 16, 16, 512)
		h_state1, p_depth1 = R_3(input_cat2,layer_name('r_clstm'),KER_SZ,q_output_channels) 
		# print('h_state1, p_depth1')
		# print(h_state1.shape,p_depth1.shape) #(1, 16, 16, 512) (1, 16, 16, 1)
		# print(3/0)
		return h_state1,c_state1,p_depth1	
	
	# def R_CLSTM(f_d,h_state,c_state): 
	# 	input_cat1 = tf.concat([f_d, h_state], 3)
	# 	F_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,256,bn=False,training=False,sigmoid=True)   
	# 	I_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,256,bn=False,training=False,sigmoid=True)   
	# 	C_t1=conv_layer_tanh(input_cat1,layer_name('conv'),KER_SZ,256,bn=False,training=False,tanh=True)      
	# 	Q_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,512,bn=False,training=False,sigmoid=True)   
	# 	c_state1 = F_t1 * c_state + I_t1 * C_t1 
	# 	# print('c_state1, Q_t1')
	# 	# print(c_state1.shape, Q_t1.shape) #(1, 16, 16, 256) (1, 16, 16, 512)
	# 	h_state1, p_depth1 = R_3(tf.concat([c_state1, Q_t1], 3)) 
	# 	# print('h_state1, p_depth1')
	# 	# print(h_state1.shape,p_depth1.shape) #(1, 16, 16, 512) (1, 16, 16, 1)
	# 	# print(3/0)
	# 	return h_state1,c_state1,p_depth1
	
	def R_CLSTM_1(f_d1,f_d2,f_d3,f_d4,f_d5): #f_d.shape=[10,256,256,1024]
		d,h, w,c = f_d1.get_shape().as_list() 
		# print('R_CLSTM_1 net:',d,h,w,c) 
		h_state_init = tf.zeros([1, h, w, 256])
		c_state_init = tf.zeros([1, h, w, 256])
		h_state, c_state = h_state_init, c_state_init
		h_state1,c_state1,_ = R_CLSTM(f_d1, h_state,c_state)  #1
		h_state5,c_state5,_ = R_CLSTM(f_d5, h_state,c_state)  #5
		h_state4,c_state4,_ = R_CLSTM(f_d4, h_state5,c_state5)  #5,4
		h_state3,c_state3,_ = R_CLSTM(f_d3, h_state4,c_state4)  #5,4,3
		h_state2,c_state2,_ = R_CLSTM(f_d2, h_state3,c_state3)  #5,4,3,2
		h_state11,c_state11,_ = R_CLSTM(f_d1, h_state2,c_state2)  #5,4,3,2
		return h_state1,h_state11
	
	def R_CLSTM_2(f_d1,f_d2,f_d3,f_d4,f_d5): #f_d.shape=[10,256,256,1024]
		d,h, w,c = f_d1.get_shape().as_list() 
		# print('R_CLSTM_2 net:',d,h,w,c) 
		h_state_init = tf.zeros([1, h, w, 256])
		c_state_init = tf.zeros([1, h, w, 256])
		h_state, c_state = h_state_init, c_state_init
		h_state1,c_state1,_ = R_CLSTM(f_d1, h_state,c_state)  #1
		h_state2,c_state2,_ = R_CLSTM(f_d2, h_state1,c_state1)  #1,2
		h_state5,c_state5,_ = R_CLSTM(f_d5, h_state,c_state)  #5
		h_state4,c_state4,_ = R_CLSTM(f_d4, h_state5,c_state5)  #5,4
		h_state3,c_state3,_ = R_CLSTM(f_d3, h_state4,c_state4)  #5,4,3
		h_state22,c_state22,_ = R_CLSTM(f_d2, h_state3,c_state3)  #5,4,3,2
		return h_state2,h_state22
	
	def R_CLSTM_3(f_d1,f_d2,f_d3,f_d4,f_d5): #f_d.shape=[10,256,256,1024]
		d,h, w,c = f_d1.get_shape().as_list() 
		# print('R_CLSTM_3 net:',d,h,w,c) 
		h_state_init = tf.zeros([1, h, w, 256])
		c_state_init = tf.zeros([1, h, w, 256])
		h_state, c_state = h_state_init, c_state_init
		h_state1,c_state1,_ = R_CLSTM(f_d1, h_state,c_state)  #1
		h_state2,c_state2,_ = R_CLSTM(f_d2, h_state1,c_state1)  #1,2
		h_state3,c_state3,_ = R_CLSTM(f_d3, h_state2,c_state2)  #1,2,3
		h_state5,c_state5,_ = R_CLSTM(f_d5, h_state,c_state)  #5
		h_state4,c_state4,_ = R_CLSTM(f_d4, h_state5,c_state5)  #5,4
		h_state33,c_state33,_ = R_CLSTM(f_d3, h_state4,c_state4)  #5,4,3
		return h_state3,h_state33	
	
	def R_CLSTM_4(f_d1,f_d2,f_d3,f_d4,f_d5): #f_d.shape=[10,256,256,1024]
		d,h, w,c = f_d1.get_shape().as_list() 
		# print('R_CLSTM_4 net:',d,h,w,c) 
		h_state_init = tf.zeros([1, h, w, 256])
		c_state_init = tf.zeros([1, h, w, 256])
		h_state, c_state = h_state_init, c_state_init
		h_state1,c_state1,_ = R_CLSTM(f_d1, h_state,c_state)  #1
		h_state2,c_state2,_ = R_CLSTM(f_d2, h_state1,c_state1)  #1,2
		h_state3,c_state3,_ = R_CLSTM(f_d3, h_state2,c_state2)  #1,2,3
		h_state4,c_state4,_ = R_CLSTM(f_d4, h_state3,c_state3)  #1,2,3,4
		h_state5,c_state5,_= R_CLSTM(f_d5, h_state,c_state)  #5
		h_state44,c_state44,_ = R_CLSTM(f_d4, h_state5,c_state5)  #5,4
		return h_state4,h_state44	
	
	def R_CLSTM_5(f_d1,f_d2,f_d3,f_d4,f_d5): #f_d.shape=[10,256,256,1024]
		d,h, w,c = f_d1.get_shape().as_list() 
		# print('R_CLSTM_5 net:',d,h,w,c) 
		h_state_init = tf.zeros([1, h, w, 256])
		c_state_init = tf.zeros([1, h, w, 256])
		h_state, c_state = h_state_init, c_state_init
		h_state1,c_state1,_ = R_CLSTM(f_d1, h_state,c_state)  #1
		h_state2,c_state2,_ = R_CLSTM(f_d2, h_state1,c_state1)  #1,2
		h_state3,c_state3,_ = R_CLSTM(f_d3, h_state2,c_state2)  #1,2,3
		h_state4,c_state4,_ = R_CLSTM(f_d4, h_state3,c_state3)  #1,2,3,4
		h_state5,c_state5,_ = R_CLSTM(f_d5, h_state4,c_state4)  #1,2,3,4
		h_state55,c_state55,_ = R_CLSTM(f_d5, h_state,c_state)  #5
		return h_state5,h_state55	
	
	def Bidi_CLSTM():#f_d.shape=[10,256,256,64]
		c11,c12,c13,c14,f_d1 = hourglass_stack_fused_depth_prediction(netIN_1)
		c21,c22,c23,c24,f_d2 = hourglass_stack_fused_depth_prediction(netIN_2)
		c31,c32,c33,c34,f_d3 = hourglass_stack_fused_depth_prediction(netIN_3)
		c41,c42,c43,c44,f_d4 = hourglass_stack_fused_depth_prediction(netIN_4)
		c51,c52,c53,c54,f_d5 = hourglass_stack_fused_depth_prediction(netIN_5)
		# print('c11.shape,f_d1.shape')
		# print(c11.shape,c12.shape,c13.shape,c14.shape) #(?, 256, 256, 64) (?, 128, 128, 128) (?, 64, 64, 256) (?, 32, 32, 512)
		# print(f_d1.shape) #(?, 16, 16, 1024)
		f1f,f1b=R_CLSTM_1(f_d1,f_d2,f_d3,f_d4,f_d5)
		f2f,f2b=R_CLSTM_2(f_d1,f_d2,f_d3,f_d4,f_d5)
		f3f,f3b=R_CLSTM_3(f_d1,f_d2,f_d3,f_d4,f_d5)
		f4f,f4b=R_CLSTM_4(f_d1,f_d2,f_d3,f_d4,f_d5)
		f5f,f5b=R_CLSTM_5(f_d1,f_d2,f_d3,f_d4,f_d5)
		# print('forword feature,backward feature')
		# print(f1f.shape,f1b.shape) #(1, 16, 16, 512) (1, 16, 16, 512)
		# print(3/0)
		a1,F1=attention(f1f,f1b);out1=fusion(c11,c12,c13,c14,F1)
		a2,F2=attention(f2f,f2b);out2=fusion(c21,c22,c23,c24,F2)
		a3,F3=attention(f3f,f3b);out3=fusion(c31,c32,c33,c34,F3)
		a4,F4=attention(f4f,f4b);out4=fusion(c41,c42,c43,c44,F4)
		a5,F5=attention(f5f,f5b);out5=fusion(c51,c52,c53,c54,F5)
		# print('--------------------name---------------')
		# print('a1',a1.name,'F1',F1.name,'out1',out1.name)	
		# print(3/0)		
		# print('F1.shape')
		# print(F1.shape) #(1, 16, 16, 1024)
		# print(3/0)
		return out1,out2,out3,out4,out5	
	
	out1,out2,out3,out4,out5=Bidi_CLSTM()
	
	return out1,out2,out3,out4,out5
