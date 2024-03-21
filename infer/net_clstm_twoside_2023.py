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

def dynamic_rnn(rnn_type='lstm'):
	# cell = tf.nn.rnn_cell.LSTMCell(state_size)
	# tf.contrib.cudnn_rnn.CudnnLSTM
	# tf.compat.v1.nn.rnn_cell.GRUCell  tf.contrib.cudnn_rnn.CudnnGRU
	# 创建输入数据,3代表batch size,6代表输入序列的最大步长(max time),4代表每个序列的维度
	X = np.random.randn(3, 6, 4)
    # 第二个输入的实际长度为4
	X[1, 4:] = 0
    #记录三个输入的实际步长
	X_lengths = [6, 4, 6]
	rnn_hidden_size = 5
	if rnn_type == 'lstm':
		cell=tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
	else:
		cell=tf.contrib.rnn.GRUCell(num_units=rnn_hidden_size)
	outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        sequence_length=X_lengths,
        inputs=X)
	# cell类型为LSTM :input.shape=[3,6,4]  output.shape=[3,6,5]  state.shape=[2,3,5]
	# cell类型为GRU :input.shape=[3,6,4]  output.shape=[3,6,5]  state.shape=[3,5]
	batch_size=10;embedding_dim=300
	inputs=tf.Variable(tf.random_normal([batch_size,embedding_dim]))
	previous_state=(tf.Variable(tf.random_normal([batch_size,128])),tf.Variable(tf.random_normal([batch_size,128])))
	lstmcell=tf.nn.rnn_cell.LSTMCell(128)
	outputs,(h_state,c_state)=lstmcell(inputs,previous_state)

def maps_2_cubes(x, b, d):
	#x.shape=[10,256,256,64]
	x1=x[0:5,...];x11=tf.expand_dims(x1,0)
	x2=x[5:,...];x22=tf.expand_dims(x2,0)
	x2_var=tf.concat([x11,x22],0)
	return x2_var

def hourglass_refinement(netIN_1,netIN_2,netIN_3,netIN_4,netIN_5,training):
	print('-'*30)
	print('Clstm Hourglass Architecture')
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
	
	def feature_fusion(f1,f2,f3,f4,f5):
		f11 = tf.concat([f2,f3],3) #1*128*256*256
		c1 = conv_layer(f11,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training) # 1*64*256*256
		cc1= conv_layer(c1,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training) # 1*64*256*256
		
		f22 = tf.concat([f4,f5],3)	#1*128*256*256
		c2 = conv_layer(f22,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		cc2 = conv_layer(c2,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		
		f33 = tf.concat([cc1,cc2],3) #1*128*256*256
		c3 = conv_layer(f33,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		cc3 = conv_layer(c3,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		
		f = tf.concat([cc3,f1],3)
		c4 = conv_layer(f,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		cc4 = conv_layer(c4,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)

		return cc4

	def generate_map(f1,f2,f3,f4,f5):
		f_3=feature_fusion(f1,f2,f3,f4,f5)
		stack_out_d = conv_layer(f_3, layer_name('conv'), 1, 1, bn=False, training=training, relu=False)
		return stack_out_d

	def R_3(input_cat):
		x0=conv_layer_nobias(input_cat,layer_name('conv'),3,72,bn=bn,training=training)
		x1=conv_layer_nobias(x0,layer_name('conv'),3,72,bn=bn,training=training)
		h=conv_layer(x1,layer_name('conv'),KER_SZ,8,bn=False,relu=False)
		pred_depth=conv_layer(x1,layer_name('conv'),1,1,bn=False,relu=False)
		return h, pred_depth

	def R_CLSTM_5_(f_d,h_state,c_state): #f_d.shape=[10,256,256,64]
		input_cat1 = tf.concat([f_d5, h_state], 3)
		F_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		I_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		C_t1=conv_layer_tanh(input_cat1,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		Q_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		c_state1 = F_t1 * c_state + I_t1 * C_t1 #c_state.shape [2,256,256,8]
		h_state1, p_depth1 = R_3(tf.concat([c_state1, Q_t1], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]
		return h_state1,c_state1,p_depth1

	def R_CLSTM_5_reverse(f_d1,f_d2,f_d3,f_d4,f_d5): #f_d.shape=[10,256,256,64]
		# input_tensor = maps_2_cubes(f_d, b, d) #[2,5,256,256,64],bsize=2
		d,h, w,c = f_d1.get_shape().as_list() 
		print('net:',d,h,w,c) 

		h_state_init = tf.zeros([1, h, w, 8])
		c_state_init = tf.zeros([1, h, w, 8])

		h_state, c_state = h_state_init, c_state_init
		h_state1,c_state1,p_depth11 = R_CLSTM_5_(f_d1, h_state,c_state)  #1
		h_state2,c_state2,p_depth12 = R_CLSTM_5_(f_d2, h_state1,c_state1) #1,2
		h_state3,c_state3,p_depth13 = R_CLSTM_5_(f_d3, h_state2,c_state2) #1,2,3
		h_state4,c_state4,p_depth14 = R_CLSTM_5_(f_d4, h_state3,c_state3) #1,2,3,4
		h_state5,c_state5,p_depth5 = R_CLSTM_5_(f_d5, h_state4,c_state4) #1,2,3,4,5 -> p_depth5为第 5 张图片的特征

		h_state6,c_state6,p_depth21 = R_CLSTM_5_(f_d5, h_state3,c_state3) #1,2,3,5 
		h_state7,c_state7,p_depth4 = R_CLSTM_5_(f_d4, h_state6,c_state6) #1,2,3,5,4 -> p_depth4为第 4 张图片的特征

		h_state8,c_state8,p_depth31 = R_CLSTM_5_(f_d4, h_state2,c_state2) #1,2,4 
		h_state9,c_state9,p_depth32 = R_CLSTM_5_(f_d5, h_state8,c_state8) #1,2,4,5
		h_state10,c_state10,p_depth3 = R_CLSTM_5_(f_d3, h_state9,c_state9) #1,2,4,5,3 -> p_depth3为第 3 张图片的特征

		h_state11,c_state11,p_depth41 = R_CLSTM_5_(f_d3, h_state1,c_state1) #1,3
		h_state12,c_state12,p_depth42 = R_CLSTM_5_(f_d4, h_state11,c_state11) #1,3,4
		h_state13,c_state13,p_depth43 = R_CLSTM_5_(f_d5, h_state12,c_state12) #1,3,4,5
		h_state14,c_state14,p_depth2 = R_CLSTM_5_(f_d2, h_state13,c_state13) #1,3,4,5,2 -> p_depth2为第 2 张图片的特征

		h_state15,c_state15,p_depth51 = R_CLSTM_5_(f_d2, h_state,c_state)  #2
		h_state16,c_state16,p_depth52 = R_CLSTM_5_(f_d3, h_state15,c_state15) #2,3
		h_state17,c_state17,p_depth53 = R_CLSTM_5_(f_d4, h_state16,c_state16) #2,3,4
		h_state18,c_state18,p_depth54 = R_CLSTM_5_(f_d5, h_state17,c_state17) #2,3,4,5
		h_state19,c_state19,p_depth1 = R_CLSTM_5_(f_d1, h_state18,c_state18) #2,3,4,5,1 -> p_depth5为第 5 张图片的特征

		return p_depth1,p_depth2,p_depth3,p_depth4,p_depth5

	def R_CLSTM_5(f_d1,f_d2,f_d3,f_d4,f_d5): #f_d.shape=[10,256,256,64]
		# input_tensor = maps_2_cubes(f_d, b, d) #[2,5,256,256,64],bsize=2
		d,h, w,c = f_d1.get_shape().as_list() 
		print('net:',d,h,w,c)
		# re_h_s1,re_h_s2,re_h_s3,re_h_s4,re_h_s5,re_c_d1,re_c_d2,re_c_d3,re_c_d4,re_c_d5=R_CLSTM_5_reverse(f_d1,f_d2,f_d3,f_d4,f_d5)
		h_state_init = tf.zeros([1, h, w, 8])
		c_state_init = tf.zeros([1, h, w, 8])
		# seq_len = 5
		h_state, c_state = h_state_init, c_state_init
		# output_inner = variable('out_10', [2, 5,h, w, 8], tf.constant_initializer(0.0))
		# output_inner = tf.zeros([2, 5,h, w, 8])
		input_cat1 = tf.concat([f_d1, h_state], 3)
		F_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		I_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		C_t1=conv_layer_tanh(input_cat1,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		Q_t1=conv_layer_sigmoid(input_cat1,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		c_state1 = F_t1 * c_state + I_t1 * C_t1 #c_state.shape [2,256,256,8]
		h_state1, p_depth1 = R_3(tf.concat([c_state1, Q_t1], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]
        
		input_cat2 = tf.concat([f_d2, h_state1], 3)
		F_t2=conv_layer_sigmoid(input_cat2,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		I_t2=conv_layer_sigmoid(input_cat2,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		C_t2=conv_layer_tanh(input_cat2,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		Q_t2=conv_layer_sigmoid(input_cat2,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		c_state2 = F_t2 * c_state1 + I_t2 * C_t2 #c_state.shape [2,256,256,8]
		h_state2, p_depth2 = R_3(tf.concat([c_state2, Q_t2], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]
        
		input_cat3 = tf.concat([f_d3, h_state2], 3)
		F_t3=conv_layer_sigmoid(input_cat3,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		I_t3=conv_layer_sigmoid(input_cat3,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		C_t3=conv_layer_tanh(input_cat3,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		Q_t3=conv_layer_sigmoid(input_cat3,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		c_state3 = F_t3 * c_state2 + I_t3 * C_t3 #c_state.shape [2,256,256,8]
		h_state3, p_depth3 = R_3(tf.concat([c_state3, Q_t3], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]
        
		input_cat4 = tf.concat([f_d4, h_state3], 3)
		F_t4=conv_layer_sigmoid(input_cat4,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		I_t4=conv_layer_sigmoid(input_cat4,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		C_t4=conv_layer_tanh(input_cat4,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		Q_t4=conv_layer_sigmoid(input_cat4,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		c_state4 = F_t4 * c_state3 + I_t4 * C_t4 #c_state.shape [2,256,256,8]
		h_state4, p_depth4 = R_3(tf.concat([c_state4, Q_t4], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]
        
		input_cat5 = tf.concat([f_d5, h_state4], 3)
		F_t5=conv_layer_sigmoid(input_cat5,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # F_t.shape [2,256,256,8]
		I_t5=conv_layer_sigmoid(input_cat5,layer_name('conv'),KER_SZ,8,bn=False,training=False,sigmoid=True)   # I_t.shape [2,256,256,8]
		C_t5=conv_layer_tanh(input_cat5,layer_name('conv'),KER_SZ,8,bn=False,training=False,tanh=True)      # C_t.shape [2,256,256,8]
		Q_t5=conv_layer_sigmoid(input_cat5,layer_name('conv'),KER_SZ,64,bn=False,training=False,sigmoid=True)   # Q_t.shape [2,256,256,8]      # C_t.shape [2,256,256,8]
		c_state5 = F_t5 * c_state4 + I_t5 * C_t5 #c_state.shape [2,256,256,8]
		h_state5, p_depth5 = R_3(tf.concat([c_state5, Q_t5], 3))     # tf.concat([c_state, Q_t], 3).shape  [2,256,256,72]                              # p_depth.shape [2,256,256,1]

		return p_depth1,p_depth2,p_depth3,p_depth4,p_depth5
	
	f_d1 = hourglass_stack_fused_depth_prediction(netIN_1)
	f_d2 = hourglass_stack_fused_depth_prediction(netIN_2)
	f_d3 = hourglass_stack_fused_depth_prediction(netIN_3)
	f_d4 = hourglass_stack_fused_depth_prediction(netIN_4)
	f_d5 = hourglass_stack_fused_depth_prediction(netIN_5)
    
	out1,out2,out3,out4,out5=R_CLSTM_5_reverse(f_d1,f_d2,f_d3,f_d4,f_d5)

    # out=R_CLSTM_5(f_d1,f_d2,f_d2,f_d2,f_d2)      # [2,5,256,256,1]
	# print('out1.shape',out1.shape)
	# x1=tf.expand_dims(p_depth,1)
	# out1_var=tf.concat([out[0,...],out[1,...]],0)
	# print('out1_var.shape',out1_var.shape)
	# out=tf.reshape(out,[10,256,256,1])   #out是所有输出的
	# out1_var = variable('out_10', [10,256,256,1], tf.constant_initializer(0.0))
	# out1=tf.zeros([10,256,256,1],dtype=tf.float32)
	# out1_var=tf.Variable(out1)  #只有tf.Variable(out1) 设置成可变变量才可以进行张量的赋值
	# out1_var[0:5,...].assign(out[0,...])
	# out1_var[5:10,...].assign(out[1,...])
	return out1,out2,out3,out4,out5
