## **********************  import **********************
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.filterwarnings('ignore')
from operator import concat, delitem
from re import T
from sre_parse import FLAGS
# import matplotlib.pyplot as plt
import tensorflow as tf
# print(tf.test.is_gpu_available())
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from os import path
import numpy as np
# import skimage.data
from PIL import Image, ImageDraw, ImageFont
import random
import scipy.misc
import math
import sys
import random
import argparse
import cv2 as cv,cv2

from MaskRCNN_Keypoint_Demo.main import main,get_keyjoins,load_model
from tensorflow.python.platform import gfile


# from utils.bidiraction_net import hourglass_refinement #这是按照之前的CLSTM网络来进行训练的 双向clstm+Attention+fusion+order
from utils.bidiraction_modify_clstm_net import hourglass_refinement #这是修改之前的CLSTM网络来进行训练的 双向clstm+Attention+fusion+order
# from utils.bidiraction_modify_clstm_net_new import hourglass_refinement #这是修改之前的CLSTM网络来进行训练的 双向clstm+Attention+fusion+order
# from utils.bid_modify_clstm_noattention_self import hourglass_refinement

from utils.IO_temp import  get_camera , save_prediction_png
from utils.Loss_functions_surr import calc_loss_normal2, calc_loss, calc_loss_d_refined_mask
from utils.Geometry_MB_surr import dmap_to_nmap
from utils.denspose_transform_functions_surr import compute_dp_tr_3d_2d_loss2 
from utils.handle_csv import calc_tk_joins

print("You are using tensorflow version ",tf.VERSION)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
## ********************** change your variables **********************
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 1
ITERATIONS = 100000000

# pre_ck_pnts_dir = "../model/depth_prediction"
# pre_ck_pnts_dir = "/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230220_clstm_attention_fusion_order/model/HDNet"
# pre_ck_pnts_dir = "/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230220_modify_clstm_attention_fusion_order_two/model/HDNet"
pre_ck_pnts_dir = "/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230330_modify_clstm_bid_two_3_2/model/HDNet"
# pre_ck_pnts_dir = "/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230330_old_modify_clstm_bid_two_2_2/model/HDNet"
# pre_ck_pnts_dir = "/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/noattention_20230330_old_modify_clstm_bid_two_3_2/model/HDNet"

model_num = '40000'
model_num_int = 40000
pre_ck_pnts_dir_Order="/data/liqianlin/HDNET_code/training/order_train/training_progress_order/model/orderlEstimator"
model_num_Order = '100000'
epoch_num='6'
surr_path = "/data/liqianlin/surreal_HDNET"
tk_path = "/data/liqianlin/Tiktok"
origin1n, scaling1n, C1n, cen1n, K1n, Ki1n, M1n, R1n, Rt1n = get_camera(BATCH_SIZE,IMAGE_HEIGHT)

ff=open(tk_path +'/correspondences/corr_mat.txt')
linef=ff.readline()
data_listf=[]
while linef:
    numf =list(map(str,linef.split(',')))
    if '\n' in numf[4]:
        numf[4]=numf[4].split('\n')[0]
    data_listf.append(numf)
    linef = ff.readline()
ff.close()
corr_mat = np.array(data_listf)

f=open("/data/liqianlin/surreal_HDNET/surreal_5_frames_dp.txt")
line=f.readline()
data_list=[]
while line:
    num =list(map(str,line.split(',')))
    if '\n' in num[4]:
        num[4]=num[4].split('\n')[0]
    data_list.append(num)
    line = f.readline()
f.close()
surreal_mat = np.array(data_list)

tiktok=len(corr_mat)
print(tiktok,len(surreal_mat))

n_tiktok=len(corr_mat)
corr_mata=np.random.permutation(corr_mat) #将数组行数打乱
surreal_mata=np.random.permutation(surreal_mat)
## **************************** define the network ****************************
refineNet_graph = tf.Graph()
with refineNet_graph.as_default():
    
    ## ****************************Order****************************
    keypoint = tf.placeholder(tf.int32, shape=(None,17,2))
    keypoint_surr = tf.placeholder(tf.int32, shape=(None,17,2))
    
    x1_tk_order = tf.placeholder(tf.float32, shape=(256,256)) #order网络来估计关节点深度
    x2_tk_order = tf.placeholder(tf.float32, shape=(256,256))
    x3_tk_order = tf.placeholder(tf.float32, shape=(256,256))
    x4_tk_order = tf.placeholder(tf.float32, shape=(256,256))
    x5_tk_order = tf.placeholder(tf.float32, shape=(256,256))
    
    ## ****************************SURREAL****************************
    x1_surr = tf.placeholder(tf.float32, shape=(None, 256,256,6))
    y1_surr = tf.placeholder(tf.float32, shape=(None, 256,256,1))#gt depth map
    z1_surr = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    n1_surr = tf.placeholder(tf.float32, shape=(None, 256,256,3))

    x2_surr = tf.placeholder(tf.float32, shape=(None, 256,256,6))
    y2_surr = tf.placeholder(tf.float32, shape=(None, 256,256,1))#gt depth map
    z2_surr = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    n2_surr = tf.placeholder(tf.float32, shape=(None, 256,256,3))

    x3_surr = tf.placeholder(tf.float32, shape=(None, 256,256,6))
    y3_surr = tf.placeholder(tf.float32, shape=(None, 256,256,1))#gt depth map
    z3_surr = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    n3_surr = tf.placeholder(tf.float32, shape=(None, 256,256,3))

    x4_surr = tf.placeholder(tf.float32, shape=(None, 256,256,6))
    y4_surr = tf.placeholder(tf.float32, shape=(None, 256,256,1))#gt depth map
    z4_surr = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    n4_surr = tf.placeholder(tf.float32, shape=(None, 256,256,3))

    x5_surr = tf.placeholder(tf.float32, shape=(None, 256,256,6))
    y5_surr = tf.placeholder(tf.float32, shape=(None, 256,256,1))#gt depth map
    z5_surr = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    n5_surr = tf.placeholder(tf.float32, shape=(None, 256,256,3))

    ## ****************************tiktok****************************
    x1_tk = tf.placeholder(tf.float32, shape=(None, 256,256,6))
    n1_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z1_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))

    x2_tk = tf.placeholder(tf.float32, shape=(None, 256,256,6)) 
    n2_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z2_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))

    x3_tk = tf.placeholder(tf.float32, shape=(None, 256,256,6)) 
    n3_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z3_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    x4_tk = tf.placeholder(tf.float32, shape=(None, 256,256,6)) 
    n4_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z4_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    x5_tk = tf.placeholder(tf.float32, shape=(None, 256,256,6)) 
    n5_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z5_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))

    
    i_r1_c1_r2_c2_1 = tf.placeholder(tf.int32, shape=(None, None,5))
    i_limit_1 = tf.placeholder(tf.int32, shape=(None, 24,3))
    i_r1_c1_r2_c2_2 = tf.placeholder(tf.int32, shape=(None, None,5))
    i_limit_2 = tf.placeholder(tf.int32, shape=(None, 24,3))
    i_r1_c1_r2_c2_3 = tf.placeholder(tf.int32, shape=(None, None,5))
    i_limit_3 = tf.placeholder(tf.int32, shape=(None, 24,3))
    i_r1_c1_r2_c2_4 = tf.placeholder(tf.int32, shape=(None, None,5))
    i_limit_4 = tf.placeholder(tf.int32, shape=(None, 24,3))
    ## *****************************camera***********************************
    R = tf.placeholder(tf.float32, shape=(3,3))
    Rt = tf.placeholder(tf.float32, shape=(3,3))
    K = tf.placeholder(tf.float32, shape=(3,3))
    Ki = tf.placeholder(tf.float32, shape=(3,3))
    C = tf.placeholder(tf.float32, shape=(3,4))
    cen = tf.placeholder(tf.float32, shape=(3))
    origin = tf.placeholder(tf.float32, shape=(None, 2))
    scaling = tf.placeholder(tf.float32, shape=(None, 1))
    
    ## ****************************Network****************************
    #将surr数据集中的color x2放入网络中预测得到 预测的深度图out_surr
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_surr1,out2_surr2,out2_surr3,out2_surr4,out2_surr5= hourglass_refinement(x1_surr,x2_surr,x3_surr,x4_surr,x5_surr,True)

    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_1_tk,out2_2_tk,out2_3_tk,out2_4_tk,out2_5_tk = hourglass_refinement(x1_tk,x2_tk,x3_tk,x4_tk,x5_tk,True)
             
    ## ****************************Loss SURREAL****************************
    loss_relation_joins_1=calc_tk_joins(keypoint[0,...],x1_tk_order,out2_1_tk)
    loss_relation_joins_2=calc_tk_joins(keypoint[1,...],x2_tk_order,out2_2_tk)
    loss_relation_joins_3=calc_tk_joins(keypoint[2,...],x3_tk_order,out2_3_tk)
    loss_relation_joins_4=calc_tk_joins(keypoint[3,...],x4_tk_order,out2_4_tk)
    loss_relation_joins_5=calc_tk_joins(keypoint[4,...],x5_tk_order,out2_5_tk)

    loss_relation_joins_11=calc_tk_joins(keypoint_surr[0,...],y1_surr[0,:,:,0],out2_surr1)
    loss_relation_joins_22=calc_tk_joins(keypoint_surr[1,...],y2_surr[0,:,:,0],out2_surr2)
    loss_relation_joins_33=calc_tk_joins(keypoint_surr[2,...],y3_surr[0,:,:,0],out2_surr3)
    loss_relation_joins_44=calc_tk_joins(keypoint_surr[3,...],y4_surr[0,:,:,0],out2_surr4)
    loss_relation_joins_55=calc_tk_joins(keypoint_surr[4,...],y5_surr[0,:,:,0],out2_surr5)

    total_loss_order_tk=(loss_relation_joins_1+loss_relation_joins_2+loss_relation_joins_3+loss_relation_joins_4+loss_relation_joins_5)*0.1
    total_loss_order_surr=(loss_relation_joins_11+loss_relation_joins_22+loss_relation_joins_33+loss_relation_joins_44+loss_relation_joins_55)*0.05
    total_loss_order=total_loss_order_tk+total_loss_order_surr
    # total_loss_order=total_loss_order_tk
    ## ****************************Loss SURREAL****************************
    nmap_surr1 = dmap_to_nmap(out2_surr1, Rt, R, Ki, cen, z1_surr, origin, scaling)
    total_loss1_surr_n_1 = calc_loss_normal2(nmap_surr1,n1_surr,z1_surr)
    total_loss1_d_surr_1=calc_loss(out2_surr1,y1_surr,z1_surr)
    total_loss2_d_surr_1 = calc_loss_d_refined_mask(out2_surr1,y1_surr,z1_surr)
    total_loss_surr_1=2*total_loss1_d_surr_1+total_loss1_surr_n_1+total_loss2_d_surr_1
    
    nmap_surr2 = dmap_to_nmap(out2_surr2, Rt, R, Ki, cen, z2_surr, origin, scaling)
    total_loss1_surr_n_2 = calc_loss_normal2(nmap_surr2,n2_surr,z2_surr)
    total_loss1_d_surr_2=calc_loss(out2_surr2,y2_surr,z2_surr)
    total_loss2_d_surr_2 = calc_loss_d_refined_mask(out2_surr2,y2_surr,z2_surr)
    total_loss_surr_2=2*total_loss1_d_surr_2+total_loss1_surr_n_2+total_loss2_d_surr_2
    
    nmap_surr3 = dmap_to_nmap(out2_surr3, Rt, R, Ki, cen, z3_surr, origin, scaling)
    total_loss1_surr_n_3 = calc_loss_normal2(nmap_surr3,n3_surr,z3_surr)
    total_loss1_d_surr_3=calc_loss(out2_surr3,y3_surr,z3_surr)
    total_loss2_d_surr_3 = calc_loss_d_refined_mask(out2_surr3,y3_surr,z3_surr)
    total_loss_surr_3=2*total_loss1_d_surr_3+total_loss1_surr_n_3+total_loss2_d_surr_3
    
    nmap_surr4 = dmap_to_nmap(out2_surr4, Rt, R, Ki, cen, z4_surr, origin, scaling)
    total_loss1_surr_n_4 = calc_loss_normal2(nmap_surr4,n4_surr,z4_surr)
    total_loss1_d_surr_4=calc_loss(out2_surr4,y4_surr,z4_surr)
    total_loss2_d_surr_4 = calc_loss_d_refined_mask(out2_surr4,y4_surr,z4_surr)
    total_loss_surr_4=2*total_loss1_d_surr_4+total_loss1_surr_n_4+total_loss2_d_surr_4
    
    nmap_surr5 = dmap_to_nmap(out2_surr5, Rt, R, Ki, cen, z5_surr, origin, scaling)
    total_loss1_surr_n_5 = calc_loss_normal2(nmap_surr5,n5_surr,z5_surr)
    total_loss1_d_surr_5=calc_loss(out2_surr5,y5_surr,z5_surr)
    total_loss2_d_surr_5 = calc_loss_d_refined_mask(out2_surr5,y5_surr,z5_surr)
    total_loss_surr_5=2*total_loss1_d_surr_5+total_loss1_surr_n_5+total_loss2_d_surr_5

    total_surr_1_d=(total_loss1_d_surr_1+total_loss1_d_surr_2+total_loss1_d_surr_3+total_loss1_d_surr_4+total_loss1_d_surr_5)*2
    total_surr_2_d=total_loss2_d_surr_1+total_loss2_d_surr_2+total_loss2_d_surr_3+total_loss2_d_surr_4+total_loss2_d_surr_5
    total_loss1_surr_nn=total_loss1_surr_n_5+total_loss1_surr_n_4+total_loss1_surr_n_3+total_loss1_surr_n_2+total_loss1_surr_n_1
    total_loss_surr=total_loss_surr_5+total_loss_surr_4+total_loss_surr_3+total_loss_surr_2+total_loss_surr_1

    ## ****************************Loss TK****************************
    
    nmap1_tk = dmap_to_nmap(out2_1_tk, Rt, R, Ki, cen, z1_tk, origin, scaling)
    nmap2_tk = dmap_to_nmap(out2_2_tk, Rt, R, Ki, cen, z2_tk, origin, scaling)
    nmap3_tk = dmap_to_nmap(out2_3_tk, Rt, R, Ki, cen, z3_tk, origin, scaling)
    nmap4_tk = dmap_to_nmap(out2_4_tk, Rt, R, Ki, cen, z4_tk, origin, scaling)
    nmap5_tk = dmap_to_nmap(out2_5_tk, Rt, R, Ki, cen, z5_tk, origin, scaling)

    total_loss_n_tk = calc_loss_normal2(nmap1_tk,n1_tk,z1_tk)+calc_loss_normal2(nmap2_tk,n2_tk,z2_tk)+calc_loss_normal2(nmap4_tk,n4_tk,z4_tk)+calc_loss_normal2(nmap3_tk,n3_tk,z3_tk)+calc_loss_normal2(nmap5_tk,n5_tk,z5_tk)
    
    loss3d_1,loss2d_1,PC2p_1,PC1_2_1,loss3d_11,loss2d_11,PC1p_1,PC2_1_1 = compute_dp_tr_3d_2d_loss2(out2_1_tk,out2_2_tk,
                                                         i_r1_c1_r2_c2_1[0,...],i_limit_1[0,...],
                                                         C,R,Rt,cen,K,Ki,origin,scaling)

    loss3d_2,loss2d_2,PC2p_2,PC1_2_2,loss3d_22,loss2d_22,PC1p_2,PC2_1_2 = compute_dp_tr_3d_2d_loss2(out2_1_tk,out2_3_tk,
                                                         i_r1_c1_r2_c2_2[0,...],i_limit_2[0,...],
                                                         C,R,Rt,cen,K,Ki,origin,scaling)
                                                         
    loss3d_3,loss2d_3,PC2p_3,PC1_2_3,loss3d_33,loss2d_33,PC1p_3,PC2_1_3 = compute_dp_tr_3d_2d_loss2(out2_1_tk,out2_4_tk,
                                                         i_r1_c1_r2_c2_3[0,...],i_limit_3[0,...],
                                                         C,R,Rt,cen,K,Ki,origin,scaling)
    loss3d_4,loss2d_4,PC2p_4,PC1_2_4,loss3d_44,loss2d_44,PC1p_4,PC2_1_4 = compute_dp_tr_3d_2d_loss2(out2_1_tk,out2_5_tk,
                                                         i_r1_c1_r2_c2_4[0,...],i_limit_4[0,...],
                                                         C,R,Rt,cen,K,Ki,origin,scaling)

    total_loss_tk = total_loss_n_tk + 3*(loss3d_1+loss3d_2+loss3d_3+loss3d_4)+2*(loss3d_11+loss3d_22+loss3d_33+loss3d_44) 
    total_tk_3d=3*(loss3d_1+loss3d_2+loss3d_3+loss3d_4)+2*(loss3d_11+loss3d_22+loss3d_33+loss3d_44) 
    ## ****************************Loss all****************************
    total_loss= total_loss_tk+total_loss_surr+total_loss_order
    
    ## ****************************optimizer****************************
    train_step = tf.train.AdamOptimizer(learning_rate=0.001,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=0.1,
                                        use_locking=False,
                                        name='Adam').minimize(total_loss)

##  ********************** initialize the network **********************
sess = tf.Session(graph=refineNet_graph,config=config)
sess1=tf.Session()

with sess.as_default():
    with refineNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt.meta')
        saver.restore(sess,pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt')
        print("Model DR restored.")
   
##  ********************** make the output folders ********************** 
#设置surreal数据集的可视化文件夹
# ck_pnts_dir = "/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230220_clstm_attention_fusion_order/model/HDNet"
# log_dir = "/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230220_clstm_attention_fusion_order/"

ck_pnts_dir = "/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230330_old_modify_clstm_bid_two_3_2/model/HDNet"
log_dir = "/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230330_old_modify_clstm_bid_two_3_2/"

if not gfile.Exists(ck_pnts_dir):
    print("ck_pnts_dir created!",ck_pnts_dir)
    gfile.MakeDirs(ck_pnts_dir)
    
# if (path.exists(log_dir+"trainLog0.2.txt")):
#     os.remove(log_dir+"trainLog0.2.txt")

max_corr_points=25000

def _parse_tiktok(token1):
    def process(frm,next_frm):
        name_i = str(frm.numpy())[2:-1]
        name_j = str(next_frm.numpy())[2:-1]
        
        i_r1_c1_r2_c2_path = tk_path + '/correspondences/corrs/' + name_i+'_'+name_j+'_i_r1_c1_r2_c2.txt' 
        i_lim_path = tk_path + '/correspondences/corrs/' + name_i+'_'+name_j+'_i_limit.txt'
        i_r1_c1_r2_c2_batch=np.array(np.genfromtxt(i_r1_c1_r2_c2_path,delimiter=","))
        i_r1_c1_r2_c2= np.zeros((1,i_r1_c1_r2_c2_batch.shape[0],5),dtype='int')
        i_r1_c1_r2_c2[0,:,:]=i_r1_c1_r2_c2_batch
        batch_i_limit=np.array(np.genfromtxt(i_lim_path,delimiter=","))
        i_limit= np.zeros((1,24,3),dtype='int')
        i_limit[0,:,:]=batch_i_limit
        return i_r1_c1_r2_c2,i_limit

    def read_correspondence_dp(frm,next_frm):
        # tf.py_function将tensor类型的str转化为普通的字符串
        [i_r1_c1_r2_c2,i_limit]=tf.py_function(process,inp=[frm,next_frm], Tout=[tf.int32,tf.int32])
        return i_r1_c1_r2_c2,i_limit

    def read_tiktok_process(frm):
        name = str(frm.numpy())[2:-1]
        batch_color=np.array(Image.open(tk_path +'/color_WO_bg/'+name+'.png'))
        color = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
        color[0,:,:,:]= batch_color
        mask=np.array(Image.open(tk_path +'/mask/'+name+'.png'))
        batch_depth=np.genfromtxt(tk_path + '/ordernet_depth_txt/' + name+'_order_depth.txt')
        depth = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
        depth[0,:,:,0] = batch_depth
        Z1 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='b')
        Z1[0,:,:,0]=mask>100
        # batch_densepose=np.array(Image.open(tk_path +'/densepose/'+name+'.png'))
        # densepose = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
        # densepose[0,:,:,:] = batch_densepose
        batch_cur_normal=np.array(Image.open(tk_path +'/pred_normals_png/'+name+'.png'))
        cur_normal= np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
        cur_normal[0,:,:,:] = batch_cur_normal
        n1 = np.genfromtxt(tk_path +'/pred_normals/'+name+'_1.txt',delimiter=" ")
        n2 = np.genfromtxt(tk_path +'/pred_normals/'+name+'_2.txt',delimiter=" ")
        n3 = np.genfromtxt(tk_path +'/pred_normals/'+name+'_3.txt',delimiter=" ")
        cur_normal[...,0] = n1;
        cur_normal[...,1] = n2;
        cur_normal[...,2] = n3;
        Z1_3 = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
        Z1_3[...,0]=Z1[...,0]
        Z1_3[...,1]=Z1[...,0]
        Z1_3[...,2]=Z1[...,0]
        X1 = np.where(Z1_3,color,np.ones_like(color)*255.0)
        N1 = np.where(Z1_3,cur_normal,np.zeros_like(cur_normal))
        Y1_tk = np.where(Z1,depth,np.zeros_like(depth))
        X_1 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,6),dtype='f')
        X_1[...,0]=X1[...,0]
        X_1[...,1]=X1[...,1]
        X_1[...,2]=X1[...,2]
        X_1[...,3]=N1[...,0]
        X_1[...,4]=N1[...,1]
        X_1[...,5]=N1[...,2]
        # X_1[...,6]=densepose[...,0]
        # X_1[...,7]=densepose[...,1]
        # X_1[...,8]=densepose[...,2]
        get_color_name=name+'.png'
        return X_1, X1, N1, Z1,Y1_tk,get_color_name
    
    def read_surreal_data(surreal_random):
        name=str(surreal_random.numpy())[2:-1]
        color_name=name+'.png'
        depth_name=name+'_depth.txt'
        normal_name=name+'_normal.png'
        normal_txt_name=name
        
        # batch_densepose=np.array(Image.open(surr_path+'/densepose/'+name+'_IUV.png'))
        # densepose = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
        # densepose[0,:,:,:] = batch_densepose
        
        batch_cur_normal=np.array(Image.open(surr_path + '/normal_png/' +normal_name))
        cur_normal= np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
        cur_normal[0,:,:,:] = batch_cur_normal
        n1 = np.genfromtxt(surr_path + '/normal_txt/' + normal_txt_name+'_NORMAL_1.txt',delimiter=" ")
        n2 = np.genfromtxt(surr_path + '/normal_txt/' + normal_txt_name+'_NORMAL_2.txt',delimiter=" ")
        n3 = np.genfromtxt(surr_path + '/normal_txt/' + normal_txt_name+'_NORMAL_3.txt',delimiter=" ")
        cur_normal[...,0] = n1;
        cur_normal[...,1] = n2;
        cur_normal[...,2] = n3;
        mask=np.array(Image.open(surr_path +'/mask/'+name+'_mask.png'))
        batch_color=np.array(Image.open(surr_path + '/color_WO_bg/' + name+'_mask.png'))
        color = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
        color[0,:,:,:] = batch_color
        batch_depth=np.genfromtxt(surr_path + '/depth/' + depth_name)
        depth = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
        depth[0,:,:,0] = batch_depth
        Z1 = np.zeros((1,256,256,1),dtype='b')
        Z1[...,0] = mask > 100
        Z1_surr = np.zeros((1,256,256,3),dtype='b')
        Z1_surr[...,0]=Z1[...,0]
        Z1_surr[...,1]=Z1[...,0]
        Z1_surr[...,2]=Z1[...,0]

        Y1_surr = np.where(Z1,depth,np.zeros_like(depth))
        x1_surr = np.where(Z1_surr,color,np.ones_like(color)*255.0)
        N1 = np.where(Z1_surr,cur_normal,np.zeros_like(cur_normal))
        x2_surr =np.zeros((1,256,256,6),dtype='f')
        x2_surr[...,0]=x1_surr[...,0]
        x2_surr[...,1]=x1_surr[...,1]
        x2_surr[...,2]=x1_surr[...,2]
        x2_surr[...,3]=N1[...,0]
        x2_surr[...,4]=N1[...,1]
        x2_surr[...,5]=N1[...,2]
        # x2_surr[...,6]=densepose[...,0]
        # x2_surr[...,7]=densepose[...,1]
        # x2_surr[...,8]=densepose[...,2]
      
        return x2_surr,N1,Y1_surr,Z1,color_name
    
    def get_tk_joins(seq1):
        name1=str(seq1.numpy())[2:-1]
        key=np.zeros((5,17,2))
        key1 = np.genfromtxt(tk_path + '/key_point_17_5/' + name1+'.txt',delimiter=" ")
        key1=np.array(key1,dtype='i')
        for d in range(5):
            key[d,...]=key1[:,(d*2):((d+1)*2)]
        key = np.array(key,dtype='i')    
        return key

    def get_tk_joins_surr(seq1):
        name1=str(seq1.numpy())[2:-1]
        key=np.zeros((5,17,2))
        key1 = np.genfromtxt(surr_path + '/key_point_17_5/' + name1+'.txt',delimiter=" ")
        key1=np.array(key1,dtype='i')
        for d in range(5):
            key[d,...]=key1[:,(d*2):((d+1)*2)]
        key_surr = np.array(key,dtype='i')    
        return key_surr    

    i_r1_c1_r2_c2_1,i_limit_1=read_correspondence_dp(token1[0],token1[1])
    i_r1_c1_r2_c2_2,i_limit_2=read_correspondence_dp(token1[0],token1[2])
    i_r1_c1_r2_c2_3,i_limit_3=read_correspondence_dp(token1[0],token1[3])
    i_r1_c1_r2_c2_4,i_limit_4=read_correspondence_dp(token1[0],token1[4])
    
    [X_1, X1, N1, Z1,Y_tk1,color_name1]=tf.py_function(read_tiktok_process,inp=[token1[0]],Tout=[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.string])
    [X_2, X2, N2, Z2,Y_tk2,color_name2]=tf.py_function(read_tiktok_process,inp=[token1[1]],Tout=[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.string])
    [X_3, X3, N3, Z3,Y_tk3,color_name3]=tf.py_function(read_tiktok_process,inp=[token1[2]],Tout=[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.string])
    [X_4, X4, N4, Z4,Y_tk4,color_name4]=tf.py_function(read_tiktok_process,inp=[token1[3]],Tout=[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.string])
    [X_5, X5, N5, Z5,Y_tk5,color_name5]=tf.py_function(read_tiktok_process,inp=[token1[4]],Tout=[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.string])

    [x1_surr,N1_surr,Y1_surr,Z1_surr]=tf.py_function(read_surreal_data,inp=[token1[5]],Tout=[tf.float32,tf.float32,tf.float32,tf.float32])   
    [x2_surr,N2_surr,Y2_surr,Z2_surr]=tf.py_function(read_surreal_data,inp=[token1[6]],Tout=[tf.float32,tf.float32,tf.float32,tf.float32])   
    [x3_surr,N3_surr,Y3_surr,Z3_surr]=tf.py_function(read_surreal_data,inp=[token1[7]],Tout=[tf.float32,tf.float32,tf.float32,tf.float32])   
    [x4_surr,N4_surr,Y4_surr,Z4_surr]=tf.py_function(read_surreal_data,inp=[token1[8]],Tout=[tf.float32,tf.float32,tf.float32,tf.float32])   
    [x5_surr,N5_surr,Y5_surr,Z5_surr]=tf.py_function(read_surreal_data,inp=[token1[9]],Tout=[tf.float32,tf.float32,tf.float32,tf.float32])   

    [key]=tf.py_function(get_tk_joins,inp=[token1[0]],Tout=[tf.int32])   
    [key_surr]=tf.py_function(get_tk_joins_surr,inp=[token1[5]],Tout=[tf.int32])   

    return X_1, N1, Z1, X_2, N2, Z2,i_r1_c1_r2_c2_1,i_limit_1,x1_surr,Y1_surr,N1_surr,Z1_surr,X1,X2,color_name1,color_name2,color_name3,color_name4,color_name5,X3,X4,X5,i_r1_c1_r2_c2_2,i_limit_2,i_r1_c1_r2_c2_3,i_limit_3,i_r1_c1_r2_c2_4,i_limit_4,token1,x2_surr,Y2_surr,N2_surr,Z2_surr,x3_surr,Y3_surr,N3_surr,Z3_surr,x4_surr,Y4_surr,N4_surr,Z4_surr,x5_surr,Y5_surr,N5_surr,Z5_surr,X_3, N3, Z3, X_4, N4, Z4,X_5, N5, Z5, key,Y_tk1,Y_tk2,Y_tk3,Y_tk4,Y_tk5,key_surr


corr=np.hstack((corr_mata,surreal_mata))
dataset=tf.data.Dataset.from_tensor_slices(corr)
dataset=dataset.shuffle(buffer_size=1000)
dataset = dataset.map(map_func=_parse_tiktok, num_parallel_calls=1)
dataset=dataset.prefetch(1)
dataset=dataset.repeat()
iterator = dataset.make_one_shot_iterator()
tiktok_color_name=iterator.get_next()

# Vis_dir_tk='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230220_clstm_attention_fusion_order/vis_tk/'
# Vis_dir_surr='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230220_clstm_attention_fusion_order/vis_surr/'

Vis_dir_tk='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230330_old_modify_clstm_bid_two_3_2/vis_tk/'
Vis_dir_surr='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230330_old_modify_clstm_bid_two_3_2/vis_surr/'


if not gfile.Exists(Vis_dir_tk):
    print("Vis_dir_tk created!",Vis_dir_tk)
    gfile.MakeDirs(Vis_dir_tk)
if not gfile.Exists(Vis_dir_surr):
    print("Vis_dir_surr created!",Vis_dir_surr)
    gfile.MakeDirs(Vis_dir_surr)
for itr in range(ITERATIONS):
    token=sess1.run([tiktok_color_name])

    key=token[0][54]
    key1=token[0][60]

    (order_tk1,order_tk2,order_tk3,order_tk4,order_tk5)=token[0][55],token[0][56],token[0][57],token[0][58],token[0][59]

    (_,loss_tk_3d,loss_tk_n,loss_tk_total,
    loss_surr_1d,loss_surr_2d,loss_surr_n,loss_surr_total,
    loss_tk_order,loss_order_total,loss_total,
    pred_surr1,pred_surr2,pred_surr3,pred_surr4,pred_surr5,
    pred_tk1, pred_tk2,pred_tk3,pred_tk4,pred_tk5,) = sess.run([train_step,total_tk_3d,total_loss_n_tk,total_loss_tk,
                                                                total_surr_1_d,total_surr_2_d,total_loss1_surr_nn,total_loss_surr,
                                                                total_loss_order_tk,total_loss_order,total_loss,
                                                                out2_surr1,out2_surr2,out2_surr3,out2_surr4,out2_surr5,
                                                                out2_1_tk,out2_2_tk,out2_3_tk,out2_4_tk,out2_5_tk],        
                                                        feed_dict={x1_surr:token[0][8],y1_surr:token[0][9],n1_surr:token[0][10],z1_surr:token[0][11],
                                                                x2_surr:token[0][29],y2_surr:token[0][30],n2_surr:token[0][31],z2_surr:token[0][32],
                                                                x3_surr:token[0][33],y3_surr:token[0][34],n3_surr:token[0][35],z3_surr:token[0][36],
                                                                x4_surr:token[0][37],y4_surr:token[0][38],n4_surr:token[0][39],z4_surr:token[0][40],
                                                                x5_surr:token[0][41],y5_surr:token[0][42],n5_surr:token[0][43],z5_surr:token[0][44],
                                                                Rt:Rt1n, Ki:Ki1n,cen:cen1n, R:R1n,
                                                                origin:origin1n,scaling:scaling1n,
                                                                x1_tk:token[0][0],n1_tk:token[0][1],z1_tk:token[0][2],
                                                                x2_tk:token[0][3],n2_tk:token[0][4],z2_tk:token[0][5],
                                                                x3_tk:token[0][45],n3_tk:token[0][46],z3_tk:token[0][47],
                                                                x4_tk:token[0][48],n4_tk:token[0][49],z4_tk:token[0][50],
                                                                x5_tk:token[0][51],n5_tk:token[0][52],z5_tk:token[0][53],
                                                                i_r1_c1_r2_c2_1:token[0][6],i_r1_c1_r2_c2_2:token[0][22],i_r1_c1_r2_c2_3:token[0][24],i_r1_c1_r2_c2_4:token[0][26],
                                                                i_limit_1:token[0][7],i_limit_2:token[0][23],i_limit_3:token[0][25],i_limit_4:token[0][27],
                                                                x1_tk_order:order_tk1[0,:,:,0],x2_tk_order:order_tk2[0,:,:,0],x3_tk_order:order_tk3[0,:,:,0],x4_tk_order:order_tk4[0,:,:,0],x5_tk_order:order_tk5[0,:,:,0],
                                                                keypoint:key,keypoint_surr:key1})

    if itr%10 == 0:
        f_err = open(log_dir+"trainLog.txt","a")
        f_err.write("%d %g,%g,%g,%g\n" % (itr*5+model_num_int,loss_tk_total,loss_surr_total,loss_order_total,loss_total))
        f_err.close()
        print(" ")
        print("loss_tk_3d %g ,loss_tk_n %g ,loss_tk_total %g " % (loss_tk_3d,loss_tk_n,loss_tk_total))
        print("loss_surr_1d %g ,loss_surr_2d %g ,loss_surr_n %g ,loss_surr_total %g " % (loss_surr_1d,loss_surr_2d,loss_surr_n,loss_surr_total))
        print("loss_tk_order %g ,loss_order_total %g " % (loss_tk_order,loss_order_total))
        print("epoch : %d , iteration %3d,loss_total %g " %(int((itr+model_num_int)/n_tiktok),(itr)+model_num_int, loss_total))
    
    if itr%1000 == 0:

        save_prediction_png (pred_surr1[0,...,0],str(itr+model_num_int),token[0][11],Vis_dir_surr,str(itr),1)
        save_prediction_png (pred_surr2[0,...,0],str(itr+1+model_num_int),token[0][32],Vis_dir_surr,str(itr),1)
        save_prediction_png (pred_surr3[0,...,0],str(itr+2+model_num_int),token[0][36],Vis_dir_surr,str(itr),1)
        save_prediction_png (pred_surr4[0,...,0],str(itr+3+model_num_int),token[0][40],Vis_dir_surr,str(itr),1)
        save_prediction_png (pred_surr5[0,...,0],str(itr+4+model_num_int),token[0][44],Vis_dir_surr,str(itr),1)
        
        # tk1_name=(str(token[0][14].decode('utf-8')).split('/')[5]).split('.')[0]
        # tk2_name=(str(token[0][15].decode('utf-8')).split('/')[5]).split('.')[0]
        # tk3_name=(str(token[0][16].decode('utf-8')).split('/')[5]).split('.')[0]
        # tk4_name=(str(token[0][17].decode('utf-8')).split('/')[5]).split('.')[0]
        # tk5_name=(str(token[0][18].decode('utf-8')).split('/')[5]).split('.')[0]

        save_prediction_png (pred_tk1[0,...,0],str(itr+model_num_int),token[0][2],Vis_dir_tk,str(itr),1)
        save_prediction_png (pred_tk2[0,...,0],str(itr+1+model_num_int),token[0][5],Vis_dir_tk,str(itr),1)
        save_prediction_png (pred_tk3[0,...,0],str(itr+2+model_num_int),token[0][47],Vis_dir_tk,str(itr),1)
        save_prediction_png (pred_tk4[0,...,0],str(itr+3+model_num_int),token[0][50],Vis_dir_tk,str(itr),1)
        save_prediction_png (pred_tk5[0,...,0],str(itr+4+model_num_int),token[0][53],Vis_dir_tk,str(itr),1)
    
    if itr % 10000 == 0 and itr != 0:
        save_path = saver.save(sess,ck_pnts_dir+"/model_"+str(itr+model_num_int)+"/model_"+str(itr+model_num_int)+".ckpt")
        