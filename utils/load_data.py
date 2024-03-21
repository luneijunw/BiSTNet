## **********************  import **********************
from __future__ import absolute_import, division, print_function, unicode_literals
# from distutils.command.config import config
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
from tensorflow.python.platform import gfile

from utils.hourglass_net_depth_singleStack_surr import hourglass_refinement
from utils.IO_temp import get_renderpeople_patch, get_camera, get_tiktok_patch, write_prediction,write_matrix_depth, write_prediction_normal, save_prediction_png
from utils.Loss_functions_surr import calc_loss_normal2, calc_loss, calc_loss_d_refined_mask
from utils.Geometry_MB_surr import dmap_to_nmap
from utils.denspose_transform_functions_surr import compute_dp_tr_3d_2d_loss2 
from utils.handle_csv import calc_order,read_csv,get_surreal_patch
from utils.hourglass_net_surr_depth_singleStack_surr import hourglass_surr_refinement
# import mxnet as mx
import time 

print("You are using tensorflow version ",tf.VERSION)
os.environ["CUDA_VISIBLE_DEVICES"]="5"
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
## ********************** change your variables **********************
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 1
ITERATIONS = 100000000

# pre_ck_pnts_dir = "../model/depth_prediction"
pre_ck_pnts_dir = "/home/laihuaijing/liqianlin/HDNet_TikTok/tf1/training/training_progress_surreal/model/HDNet"
# pre_ck_pnts_dir = "/home/laihuaijing/liqianlin/HDNet_TikTok/docker/tf1/training/training_progress_docker/model/HDNet"
model_num = '0'
model_num_int = 0

rp_path = "../../../training/training_data/Tang_data"
tk_path = "../../../training/training_data/tiktok_data"
RP_image_range = range(0,188)
origin1n, scaling1n, C1n, cen1n, K1n, Ki1n, M1n, R1n, Rt1n = get_camera(BATCH_SIZE,IMAGE_HEIGHT)

## **************************** 加载csv_name,以及csv_train_point ****************************
name_csv_path="/home/laihuaijing/liqianlin/HDNet_TikTok/training/training_data/surreal_data/surreal_y_x_train_name.csv"
train_point_csv_path="/home/laihuaijing/liqianlin/HDNet_TikTok/training/training_data/surreal_data/new_surreal_y_x_train_to256.csv"
mask_surreal_path="/home/laihuaijing/liqianlin/HDNet_TikTok/training/training_data/surreal_data/mask/"
csv_file_arr,train_point_csv_arr,all_num_image,point_row=read_csv(name_csv_path,train_point_csv_path)
# print(csv_file_arr)

    
## **************************** define the network ****************************
refineNet_graph = tf.Graph()
with refineNet_graph.as_default():
    
    ## ****************************SURREAL****************************
    
    #y2设置为一个行数未知，列数为5的数组，其中表示'> < =',分别使用1，-1，0代替，所以为int类型
    x2_1 = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    # x2 = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    y2 = tf.placeholder(tf.int32, shape=(None,5))
    y3 = tf.placeholder(tf.float32, shape=(None, 256,256,1))#gt depth map
    z2 = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    # point_row_single_pic=tf.placeholder(tf.int32, shape=[])
    ## ****************************RENDERPEOPLE****************************
    
    x1 = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    y1 = tf.placeholder(tf.float32, shape=(None, 256,256,1))
    n1 = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z1 = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    
    ## ****************************tiktok****************************
    x1_tk = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    n1_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z1_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    
    x2_tk = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    n2_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z2_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    
    i_r1_c1_r2_c2 = tf.placeholder(tf.int32, shape=(None, 25000,5))
    i_limit = tf.placeholder(tf.int32, shape=(None, 24,3))
    
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
    # with tf.variable_scope('hourglass_stack_fused_surr_depth_prediction', reuse=tf.AUTO_REUSE):
    #     out_surr = hourglass_surr_refinement(x2,True)
        # print('111',type(out_surr))
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_surr = hourglass_refinement(x2_1,True)
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_1 = hourglass_refinement(x1,True)
        # print('222',type(out2_1))
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_1_tk = hourglass_refinement(x1_tk,True)
        
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_2_tk = hourglass_refinement(x2_tk,True)
        
    ## ****************************Loss SURREAL****************************
    
    # total_loss_surr = calc_order(out_surr,y2)
    total_loss_surr_2=calc_order(out2_surr,y2)
    # total_loss_d_surr=calc_loss(out2_surr,y3,z2)
    # total_loss_surr=total_loss_surr_2+total_loss_d_surr

    ## ****************************Loss RP****************************
    
    nmap1 = dmap_to_nmap(out2_1, Rt, R, Ki, cen, z1, origin, scaling)

    total_loss1_d = calc_loss(out2_1,y1,z1)
 
    total_loss2_d = calc_loss_d_refined_mask(out2_1,y1,z1)

    total_loss_n = calc_loss_normal2(nmap1,n1,z1)

    total_loss_rp = 2*total_loss1_d + total_loss2_d + total_loss_n
    
    ## ****************************Loss TK****************************
    
    nmap1_tk = dmap_to_nmap(out2_1_tk, Rt, R, Ki, cen, z1_tk, origin, scaling)
    nmap2_tk = dmap_to_nmap(out2_2_tk, Rt, R, Ki, cen, z2_tk, origin, scaling)

    total_loss_n_tk = calc_loss_normal2(nmap1_tk,n1_tk,z1_tk)+calc_loss_normal2(nmap2_tk,n2_tk,z2_tk)
    
    loss3d,loss2d,PC2p,PC1_2 = compute_dp_tr_3d_2d_loss2(out2_1_tk,out2_2_tk,
                                                         i_r1_c1_r2_c2[0,...],i_limit[0,...],
                                                         C,R,Rt,cen,K,Ki,origin,scaling)

    total_loss_tk = total_loss_n_tk + 5*loss3d

    ## ****************************Loss all****************************
    # total_loss = total_loss_rp+total_loss_tk+total_loss_surr
    total_loss_1 = total_loss_rp+total_loss_tk+total_loss_surr_2
    # total_loss_2= total_loss_rp+total_loss_tk+total_loss_surr
    
    ## ****************************optimizer****************************
    train_step = tf.train.AdamOptimizer(learning_rate=0.001,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=0.1,
                                        use_locking=False,
                                        name='Adam').minimize(total_loss_1)

##  ********************** initialize the network **********************
# config = tf.ConfigProto(allow_soft_placement=True) 
sess = tf.Session(graph=refineNet_graph,config=config)
with sess.as_default():
    with refineNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph(pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt.meta')
        # saver.restore(sess,pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt')
        # print("Model restored.")
        
##  ********************** make the output folders ********************** 
#设置surreal数据集的可视化文件夹
Vis_dir_surr  = "../training_progress_surreal/visualization/HDNet/Surreal/"

ck_pnts_dir = "../training_progress_surreal/model/HDNet"
# ck_pnts_dir="../training_randomtensor_for_testdata/model/HDNET"
Vis_dir  = "../training_progress_surreal/visualization/HDNet/tiktok/"
# Vis_dir  = "../training_randomtensor_for_testdata/visualization/HDNet/tiktok/"
log_dir = "../training_progress_surreal/"
Vis_dir_rp  = "../training_progress_surreal/visualization/HDNet/Tang/"
# log_dir = "../training_randomtensor_for_testdata/"
# Vis_dir_rp  = "../training_randomtensor_for_testdata/visualization/HDNet/Tang/"

#若不存在surreal数据集可视化的文件夹 则 创建文件夹
if not gfile.Exists(Vis_dir_surr):
    print("Vis_dir_surr created!")
    gfile.MakeDirs(Vis_dir_surr)

if not gfile.Exists(ck_pnts_dir):
    print("ck_pnts_dir created!")
    gfile.MakeDirs(ck_pnts_dir)

if not gfile.Exists(Vis_dir):
    print("Vis_dir created!")
    gfile.MakeDirs(Vis_dir)
    
if not gfile.Exists(Vis_dir_rp):
    print("Vis_dir created!")
    gfile.MakeDirs(Vis_dir_rp)
    
if (path.exists(log_dir+"trainLog.txt")):
    os.remove(log_dir+"trainLog.txt")
   
# 训练集数量
def get_train_dataset():
    tk_nums=10
    rp_nums=10
    surr_num=10
    # 生成所有下标
    tk_indexs=[x for x in range(tk_nums)]
    rp_indexs=[x for x in range(rp_nums)]
    surr_indexs=[x for x in range(surr_num)]
    #获取surral数据集中的数据（order数组，）

    (x2_surr,x1_surr, name,name_list,name_list_row,point,Y1_surr,Z1_surr) = get_surreal_patch(mask_surreal_path,train_point_csv_arr,surr_num,all_num_image,csv_file_arr,point_row)
    # print(point)

    x1_surr=x1_surr[:,np.newaxis,:,:,:]
    x2_surr=x2_surr[:,np.newaxis,:,:,:]
    Y1_surr=Y1_surr[:,np.newaxis,:,:,:]
    Z1_surr=Z1_surr[:,np.newaxis,:,:,:]

    (X_1_tk, X1_tk, N1_tk, Z1_tk, DP1_tk, Z1_3_tk, 
        X_2_tk, X2_tk, N2_tk, Z2_tk, DP2_tk, Z2_3_tk, 
        i_r1_c1_r2_c2_tks, i_limit_tks, 
        frms_tk, frms_neighbor_tk) = get_tiktok_patch(tk_path, tk_nums, IMAGE_HEIGHT,IMAGE_WIDTH)

    X_1_tk=X_1_tk[:,np.newaxis,:,:,:]
    N1_tk=N1_tk[:,np.newaxis,:,:,:]
    Z1_tk=Z1_tk[:,np.newaxis,:,:,:]
    X_2_tk=X_2_tk[:,np.newaxis,:,:,:]
    N2_tk=N2_tk[:,np.newaxis,:,:,:]
    Z2_tk=Z2_tk[:,np.newaxis,:,:,:]

    (X_1, X1, Y1, N1, 
            Z1, DP1, Z1_3,frms) = get_renderpeople_patch(rp_path, rp_nums, RP_image_range, 
                                                        IMAGE_HEIGHT,IMAGE_WIDTH)
    X_1=X_1[:,np.newaxis,:,:,:]
    Y1=Y1[:,np.newaxis,:,:,:]
    N1=N1[:,np.newaxis,:,:,:]
    Z1=Z1[:,np.newaxis,:,:,:] 

    with tf.Session() as sess1:
        #开启线程协调器
        coord = tf.train.Coordinator()
        #创建子线程去进行操作，返回线程列表
        threads = tf.train.start_queue_runners(sess1,coord = coord)
        #打印
        print(sess1.run([Z1_surr,Y1_surr,x2_surr,name_list_row,x1_surr,point,surr_indexs,tk_indexs,rp_indexs,i_limit_tks,i_r1_c1_r2_c2_tks,X_1,Y1,N1,Z1,X_1_tk,N1_tk,Z1_tk,X_2_tk,N2_tk,Z2_tk]))
        #回收
        coord.request_stop()   #强制请求线程停止
        coord.join(threads)    #等待线程终止回收
    print(3/0)
    return Z1_surr,Y1_surr,x2_surr,name_list_row,x1_surr,point,surr_indexs,tk_indexs,rp_indexs,i_limit_tks,i_r1_c1_r2_c2_tks,X_1,Y1,N1,Z1,X_1_tk,N1_tk,Z1_tk,X_2_tk,N2_tk,Z2_tk 
# tic = time.time()
# print('开始读数据：')
# name_list_row,x1_surr,point,surr_indexs,tk_indexs,rp_indexs,i_limit_tks,i_r1_c1_r2_c2_tks,X_1,Y1,N1,Z1,X_1_tk,N1_tk,Z1_tk,X_2_tk,N2_tk,Z2_tk=get_train_dataset()   
# print(point)
# print(type(point))

##  ********************** Run the training **********************     
for itr in range(ITERATIONS):
    if (itr % 10==0):
        Z1_surr,Y1_surr,x2_surr,name_list_row,x1_surr,point,surr_indexs,tk_indexs,rp_indexs,i_limit_tks,i_r1_c1_r2_c2_tks,X_1,Y1,N1,Z1,X_1_tk,N1_tk,Z1_tk,X_2_tk,N2_tk,Z2_tk=get_train_dataset()   
    print("22222222222222222222")
    print(3/0)
    tk_index=itr%10
    # tk_index=int(np.random.choice(tk_indexs,size=1))
    rp_index=int(np.random.choice(rp_indexs,size=1))
    surr_index=int(np.random.choice(surr_indexs,size=1))

    i_limit_tk=np.array(i_limit_tks[tk_index])
    i_limit_tk=i_limit_tk.astype(float)
    i_limit_tk=np.reshape(i_limit_tk,[1,24,3])
    
    i_r1_c1_r2_c2_tk=np.array(i_r1_c1_r2_c2_tks[tk_index])
    i_r1_c1_r2_c2_tk=i_r1_c1_r2_c2_tk.astype(float)
    i_r1_c1_r2_c2_tk=np.reshape(i_r1_c1_r2_c2_tk,[1,25000,5])
    print("-------------------------------------------------")
    print("tk_index ",tk_index)
    # print("-------------------------------------------------")
    
    (_,loss_val_1,prediction1,prediction2_surr,loss_surr_2,nmap1_pred, 
     prediction1_tk,nmap1_pred_tk,PC2pn,PC1_2n) = sess.run([train_step,total_loss_1,out2_1,out2_surr,total_loss_surr_2,
                                                           nmap1,out2_1_tk,nmap1_tk,PC2p,PC1_2],
                                                  feed_dict={x1:X_1[rp_index],y1:Y1[rp_index],n1:N1[rp_index],z1:Z1[rp_index],
                                                             y2:point[surr_index],x2_1:x2_surr[surr_index],y3:Y1_surr[surr_index],z2:Z1_surr[surr_index],
                                                             Rt:Rt1n, Ki:Ki1n,cen:cen1n, R:R1n,
                                                             origin:origin1n,scaling:scaling1n,
                                                             x1_tk:X_1_tk[tk_index],n1_tk:N1_tk[tk_index],z1_tk:Z1_tk[tk_index],
                                                             x2_tk:X_2_tk[tk_index],n2_tk:N2_tk[tk_index],z2_tk:Z2_tk[tk_index],
                                                             i_r1_c1_r2_c2:i_r1_c1_r2_c2_tk,
                                                             i_limit:i_limit_tk})
    
    # print(3/0)
    # write_matrix_depth(Vis_dir_rp,prediction1,itr)
    if itr%10 == 0:
        # f_err = open(log_dir+"trainLog.txt","a")
        # f_err.write("%d %g\n" % (itr+model_num_int,loss_val_1))
        # f_err.close()
        # loss_surr=np.array(loss_surr)
        print("")
        print("iteration %3d, depth refinement training loss is %g ,surreal_order loss %g ." %(itr+model_num_int,  loss_val_1, loss_surr_2))
        # print("loss-surr : ",loss_surr)
    # if itr % 100 == 0:
        # write_matrix_depth(Vis_dir_surr,prediction2_surr,itr)
    #     # visually compare the first sample in the batch between predicted and ground truth
        # fidx = [int(frms[0])]
    #     # write_prediction(Vis_dir_rp,prediction1,itr,fidx,Z1);
        # write_prediction_normal(Vis_dir_rp,nmap1_pred,itr,fidx,Z1)
    #     # save_prediction_png (prediction1[0,...,0],nmap1_pred[0,...],X1,Z1,Z1_3,Vis_dir_rp,itr,fidx,1)
    #     fidx = [int(frms_tk[0])]
    #     # write_prediction(Vis_dir,prediction1_tk,itr,fidx,Z1_tk);
    #     # write_prediction_normal(Vis_dir,nmap1_pred_tk,itr,fidx,Z1_tk)
    #     # save_prediction_png (prediction1_tk[0,...,0],nmap1_pred_tk[0,...],X1_tk,Z1_tk,Z1_3_tk,Vis_dir,itr,fidx,1)

    # if itr % 5000 == 0 and itr != 0:
    #     save_path = saver.save(sess,ck_pnts_dir+"/model_"+str(itr+model_num_int)+"/model_"+str(itr+model_num_int)+".ckpt")






