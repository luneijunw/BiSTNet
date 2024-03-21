import tensorflow as tf
# print(tf.test.is_gpu_available())#可以将GPU加入到容器中
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
# import skimage.data
from PIL import Image, ImageDraw, ImageFont
import random
import sys
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.ERROR)

import math
from tensorflow.python.platform import gfile
import scipy.misc

# from hourglass_train_5fram_9chanls import hourglass_refinement #44
# from changemodel_order import hourglass_refinement #最新的训练 0.1_0.05
# from clstm_8 import hourglass_refinement 
from net_verfity_concat import hourglass_refinement # 证明0.1_0.05模型种的网络结构是每两个帧之间特征图concat，而现在是五个帧的特征一起concat.

# from hourglass_net_surr_order_singleStack_surr import hourglass_order_refinement as hourglass_refinement
# from hourglass_net_depth import hourglass_refinement
# from net_raw import hourglass_refinement # raw_510000 没有order,raw_order_480000 0.1_0.05


from hourglass_net_normal import hourglass_normal_prediction
from utils import (write_matrix_txt,get_origin_scaling,get_concat_h, depth2mesh, get_test_data_6_channals_thuman,read_test_data_6_channals_tang, read_test_data_6_channals_thuman,nmap_normalization,get_test_data_6_channals_tang) 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
############################## test path and outpath ##################################
# data_main_path = '/data/liqianlin/HDNET_code/Evaluaton/tang_test_1018'
data_main_path = '/data/liqianlin/HDNET_code/Evaluaton/Thuman_test_1018'
# outpath = data_main_path+"/hdnet_surreal_order_1025/infer_1025000/"
# data_main_path = '/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/temp1/'
# data_main_path = '/data/liqianlin/HDNET_code/Evaluaton/newtang_5frams_tset_1270/'

visualization = True

##############################    Inference Code     ##################################
# pre_ck_pnts_dir_DR =  "../model/depth_prediction"
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/hdnet_surreal_train/train_progress_2/model/HDNet'
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/hdnet_surreal_train/train_progress/model/HDNet'
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/hdnet_surreal_order_train/train_progress/model/HDNet'
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/hdnet_surreal_train/train_progress/model/HDNet/' #1025
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20221009_train_progress_5frams_6chanls_0.1_0.05/model/HDNet/' #1009 30
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/train_progress_0.2_2_changemodel/model/HDNet' #之前 44
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20221009_train_progress_5frams_9chanls_0.1_changemodel/model/HDNet' #1009 28
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20221009_cltm_order0.1_train_progress_5frams_6chanls_8/model/HDNet' #1009 30 clstm
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20221009_cltm_order0.1_0.05_train_progress_5frams_6chanls_8/model/HDNet' #1009 30 clstm
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/hdnet_surreal_order_train/train_progress_0.1_2/model/HDNet' #84
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20221108_cltm_noorder_train_progress_5frams_6chanls_8/model/HDNet' #验证clstm0.1_0.05,没有order时的效果
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20221108_noorder_changemodel_train_progress_5frams_6chanls/model/HDNet' #验证0.1_0.05,没有order时的效果
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20221009_train_progress_5frams_9chanls_0.1_changemodel/model/HDNet'
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20221009_train_progress_5frams_raw/model/HDNet' #raw重新训练，没有order
# pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20221009_train_progress_5frams_raw_order0.1_0.05/model/HDNet'
pre_ck_pnts_dir_DR = '/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20221110_0.1_0.5_verfity_net_concat_withorder/model/HDNet'

sec_path='/0.1_0.05_verfity_concat/'
third_path='320000'
model_num_DR = third_path
outpath = data_main_path+sec_path+'infer_'+third_path+'/'


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
pre_ck_pnts_dir_NP =  "/data/liqianlin/HDNET_code/model/normal_prediction/"
model_num_NP = '1710000'

save_depth_path=data_main_path+sec_path+'depth_'+third_path+'/'
save_depth_path_white=data_main_path+sec_path+'depthwhite_'+third_path+'/'
save_normal_path=data_main_path+sec_path+'normal_3_'+third_path+'/'

# Creat the outpath if not exists
Vis_dir = outpath
if not gfile.Exists(Vis_dir):
    print("Vis_dir created!")
    gfile.MakeDirs(Vis_dir)
if not gfile.Exists(save_depth_path):
    print("Vis_dir created!")
    gfile.MakeDirs(save_depth_path)
if not gfile.Exists(save_normal_path):
    print("Vis_dir created!")
    gfile.MakeDirs(save_normal_path)
if not gfile.Exists(save_depth_path_white):
    print("save_depth_path_white created!")
    gfile.MakeDirs(save_depth_path_white)

refineNet_graph = tf.Graph()
NormNet_graph = tf.Graph()

# Define the depth and normal networks
# ***********************************Normal Prediction******************************************
with NormNet_graph.as_default():
    # config = tf.ConfigProto(allow_soft_placement=True)
    x1_n = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    with tf.variable_scope('hourglass_normal_prediction', reuse=tf.AUTO_REUSE):
        out2_normal = hourglass_normal_prediction(x1_n,True)
# ***********************************Depth Prediction******************************************
with refineNet_graph.as_default():
    # config = tf.ConfigProto(allow_soft_placement=True)
    x1 = tf.placeholder(tf.float32, shape=(None, 256,256,6))
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_1,_,_,_,_= hourglass_refinement(x1,x1,x1,x1,x1,True)
# with refineNet_graph.as_default():
#     # config = tf.ConfigProto(allow_soft_placement=True)
#     x1 = tf.placeholder(tf.float32, shape=(None, 256,256,6))
#     with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
#         out2_1 = hourglass_refinement(x1,True)
# load checkpoints
sess2 = tf.Session(graph=NormNet_graph)
sess = tf.Session(graph=refineNet_graph)
with sess.as_default():
    with refineNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(pre_ck_pnts_dir_DR+'/model_'+model_num_DR+'/model_'+model_num_DR+'.ckpt.meta')
        saver.restore(sess,pre_ck_pnts_dir_DR+'/model_'+model_num_DR+'/model_'+model_num_DR+'.ckpt')
        print("Model DR restored.",model_num_DR)
with sess2.as_default():
    with NormNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver2 = tf.train.Saver()
        saver2 = tf.train.import_meta_graph(pre_ck_pnts_dir_NP+'/model_'+model_num_NP+'/model_'+model_num_NP+'.ckpt.meta')
        saver2.restore(sess2,pre_ck_pnts_dir_NP+'/model_'+model_num_NP+'/model_'+model_num_NP+'.ckpt')
        print("Model NP restored.",model_num_NP)
        
# Read the test images and run the HDNet
# tang='/data/liqianlin/Evalation_Hdnet/new_test_tang_1310/'
# tang='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/temp/'
# tang_testdata='/data/liqianlin/HDNET_code/Evaluaton/tang_paper_png_test/gt_256'
thuman='/data/liqianlin/Evalation_Hdnet/Thuman_test_1300'
# tang='/data/liqianlin/Evalation_Hdnet/tang_eva_连续帧_1300/'
# test_files = get_test_data_6_channals_tang(tang)
test_files=os.listdir(thuman+'/color/')
# test_files = get_test_data_6_channals_tang(tang)
print('len',len(test_files))
for f in range(0,len(test_files)):
    data_name = test_files[f]
    data_name=data_name.split('.')[0] # thuman数据集
    # data_name=data_name.split('_rgb')[0] # tang数据集 
    # data_name = str(f)
    print(f,'  Processing file: ',data_name)
    X,Z, Z3, zb = read_test_data_6_channals_thuman(thuman,data_name,IMAGE_HEIGHT,IMAGE_WIDTH)
    prediction1n = sess2.run([out2_normal],feed_dict={x1_n:X})
    
    normal_pred_raw  = np.asarray(prediction1n)[0,...]
    normal_pred = nmap_normalization(normal_pred_raw)
    
    normal_pred = np.where(Z3,normal_pred,np.zeros_like(normal_pred))
    X_1 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,6),dtype='f') 

    X_1[...,0]=X[...,0]
    X_1[...,1]=X[...,1]
    X_1[...,2]=X[...,2]
    X_1[...,3]=normal_pred[...,0]
    X_1[...,4]=normal_pred[...,1]
    X_1[...,5]=normal_pred[...,2]
 
    prediction1 = sess.run([out2_1],feed_dict={x1:X_1})
    image  = np.asarray(prediction1)[0,0,...,0]
    imagen = normal_pred[0,...]
        
    write_matrix_txt(image*Z[0,...,0],Vis_dir+data_name+".txt")
    # write_matrix_txt(imagen[...,0]*Z[0,...,0],Vis_dir+data_name+"_normal_1.txt")
    # write_matrix_txt(imagen[...,1]*Z[0,...,0],Vis_dir+data_name+"_normal_2.txt")
    # write_matrix_txt(imagen[...,2]*Z[0,...,0],Vis_dir+data_name+"_normal_3.txt")
    # depth2mesh(image*Z[0,...,0], Z[0,...,0], Vis_dir+data_name+"_mesh")
    if visualization:
        depth_map = image*Z[0,...,0]
        normal_map = imagen*Z3[0,...]
        min_depth = np.amin(depth_map[depth_map>0])
        max_depth = np.amax(depth_map[depth_map>0])
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth
        
        normal_map_rgb = -1*normal_map
        normal_map_rgb[...,2] = -1*((normal_map[...,2]*2)+1)
        normal_map_rgb = np.reshape(normal_map_rgb, [256,256,3]);
        normal_map_rgb = (((normal_map_rgb + 1) / 2) * 255).astype(np.uint8);
        
        plt.imsave(save_depth_path+data_name+"_depth.png", depth_map, cmap="hot") 
        # plt.imsave(save_normal_path+data_name+"_normal.png", normal_map_rgb) #生成的normal是4通道的，但是评估的时候是用3通道的图片进行评估；
        normal_map_rgb = Image.fromarray(normal_map_rgb)
        normal_map_rgb.save(save_normal_path+data_name+"_normal.png")
        d = np.array(scipy.misc.imread(save_depth_path+data_name+"_depth.png"),dtype='f')
        d = np.where(Z3[0,...]>0,d[...,0:3],255.0);d1=Image.fromarray(np.uint8(d))
        d1.save(save_depth_path_white+data_name+"_depth.png")
        n = np.array(scipy.misc.imread(save_normal_path+data_name+"_normal.png"),dtype='f')
        n = np.where(Z3[0,...]>0,n[...,0:3],255.0)
        final_im = get_concat_h(Image.fromarray(np.uint8(X[0,...])),Image.fromarray(np.uint8(d)))
        final_im = get_concat_h(final_im,Image.fromarray(np.uint8(n)))
        final_im.save(Vis_dir+data_name+"_results.png")
    # print(3/0)
        # os.remove(Vis_dir+data_name+"_depth.png")
        # os.remove(Vis_dir+data_name+"_normal.png")