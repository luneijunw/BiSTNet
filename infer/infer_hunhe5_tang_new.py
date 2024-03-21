from genericpath import isfile
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

from bid_modify_clstm_net import hourglass_refinement #双向，修改clstm的结构


from hourglass_net_normal import hourglass_normal_prediction
from utils import (write_matrix_txt,get_origin_scaling,get_concat_h, depth2mesh, get_test_data_6_channals_thuman,read_test_data_6_channals_tang, read_test_data_6_channals_thuman,nmap_normalization,get_test_data_6_channals_tang) 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
############################## test path and outpath ##################################
data_main_path = '/data/liqianlin/HDNET_code/Evaluaton/tang_test_1018'
# data_main_path = '/data/liqianlin/HDNET_code/Evaluaton/Thuman_test_1018'
# outpath = data_main_path+"/hdnet_surreal_order_1025/infer_1025000/"
# data_main_path = '/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/temp/'
# data_main_path = '/data/liqianlin/HDNET_code/Evaluaton/newtang_5frams_tset_1270/'

visualization = False
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
f=open("/data/liqianlin/HDNET_code/selfwritecode_HDNET/tang/tang_lianxu5_clstm.txt")
line=f.readline()
data_list=[]
while line:
    num =list(map(str,line.split(',')))
    # print(num)
    if '\n' in num[4]:
        num[4]=num[4].split('\n')[0]
    data_list.append(num)
    line = f.readline()
f.close()
tang_mat = np.array(data_list)

pre_ck_pnts_dir_DR ='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/20230330_old_modify_clstm_bid_two_no_order/model/HDNet/' # old bid two 190000 noattention (系数为 3，2)

sec_path='/2023_hunhe_old_bid_modify_clstm_two_3_2_200000_no_order/'
third_path='200000'
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
    x2 = tf.placeholder(tf.float32, shape=(None, 256,256,6))
    x3 = tf.placeholder(tf.float32, shape=(None, 256,256,6))
    x4 = tf.placeholder(tf.float32, shape=(None, 256,256,6))
    x5 = tf.placeholder(tf.float32, shape=(None, 256,256,6))

    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_11,_,_,_,_ = hourglass_refinement(x1,x1,x1,x1,x1,True)
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_1,out2_2,out2_3,out2_4,out2_5 = hourglass_refinement(x1,x2,x3,x4,x5,True)

# with refineNet_graph.as_default():
#     # config = tf.ConfigProto(allow_soft_placement=True)
#     x1 = tf.placeholder(tf.float32, shape=(None, 256,256,6))
#     with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
#         out2_1 = hourglass_refinement(x1,True)
# load checkpoints
sess2 = tf.Session(graph=NormNet_graph)
sess = tf.Session(graph=refineNet_graph,config=config)
with sess.as_default():
    with refineNet_graph.as_default():
        # tf.global_variables_initializer().run()
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
tang='/data/liqianlin/Evalation_Hdnet/new_test_tang_1310/'
# tang='/data/liqianlin/HDNET_code/training/surr_tiktok_order_model/temp/'
# tang_testdata='/data/liqianlin/HDNET_code/Evaluaton/tang_paper_png_test/gt_256'
# thuman='/data/liqianlin/Evalation_Hdnet/Thuman_test_1300'
# tang='/data/liqianlin/Evalation_Hdnet/tang_eva_1300/' #连续的1300
# test_files = get_test_data_6_channals_tang(tang)
# test_files=os.listdir(tang+'color/')
# print('len',len(test_files))
m=5
ll=os.listdir(tang+'color/')
ll_dict=dict.fromkeys(ll,100.0)

def depth_err(Y1,Y_GT,num_pixel): #第一版，符合。测得的结果是（*100）：
    diff = (Y1-Y_GT)                 #1_5:8.08  1025:8.13  84:8.32  
    diff =np.abs(diff)               #192:8.83  clstm_1_5:8.36
    square_diff = diff               # yt = (yt+ 2- med_yt)
    rms_sum = float(square_diff.sum())
    loss_rms=np.mean((rms_sum / num_pixel))
    return loss_rms
cc=0
def comp_deptherr(Zz,gt_depth_path,data_name,image,Z1,Z3,imagen):
    global cc
    Z2=np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH),dtype='f')
    Z2=Z1[0,...,0]
    num_pixel=np.sum(Zz!=0)
    gt_depth1=np.loadtxt(gt_depth_path+data_name+'.txt')
    gt_depth1=gt_depth1.astype(float)
    temp_gt=gt_depth1
    Y_GT=temp_gt
    Zz[Y_GT<1.0] = False
    Zz[Y_GT>10.0] = False
    Y_GT = np.where(Zz,Y_GT,np.zeros_like(Y_GT))
    yt_GT = Y_GT
    yt_n0_GT1 = yt_GT[yt_GT>0]
    med_yt_GT = np.median(yt_n0_GT1)
    yt_GT = (yt_GT +2- med_yt_GT) 
    Y_GT = np.where(Zz,yt_GT,np.zeros_like(Y_GT))
    Y_GT=Y_GT[Zz]
    Y_GT=Y_GT.reshape(1,-1)[0]
    pre1=(image*Z2).astype(float)
    temp_pre=pre1
    Y1=temp_pre
    Y1 = np.where(Zz,Y1,np.zeros_like(Y1))
    yt=Y1
    yt_n01 = yt[yt>0]
    med_yt = np.median(yt_n01)
    yt = (yt +2- med_yt)
    Y2 = yt
    Y1=Y2 
    Y1 = np.where(Zz,Y1,np.zeros_like(Y1))
    Y1 = Y1[Zz]
    Y1=Y1.reshape(1,-1)[0]
    loss=depth_err(Y1,Y_GT,num_pixel)
    strr=data_name+'_rgb.png'
    if ll_dict[strr]>loss:
        nn=ll_dict[strr]
        ll_dict[strr]=loss
        write_matrix_txt(image*Z2,Vis_dir+data_name+".txt")
        depth_map = image*Z2
        depth_map = np.where(Zz,depth_map,np.zeros_like(depth_map))
        min_depth = np.amin(depth_map[depth_map>0])
        max_depth = np.amax(depth_map[depth_map>0])
        mean_depth= np.mean(depth_map[depth_map>0])
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth
        print('min,max,mean,max-min,max-mean,mean-min',min_depth,max_depth,mean_depth,max_depth-min_depth,max_depth-mean_depth,mean_depth-min_depth)
        # depth_map[depth_map > (depth_map-mean_depth)] = mean_depth
        plt.imsave(save_depth_path+data_name+".png", depth_map, cmap="hot") 
        d = np.array(scipy.misc.imread(save_depth_path+data_name+".png"),dtype='f')
        d = np.where(Z3[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
        d.save(save_depth_path_white+data_name+".png")
        normal_map = imagen*Z3[0,...]
        normal_map_rgb = -1*normal_map
        normal_map_rgb[...,2] = -1*((normal_map[...,2]*2)+1)
        normal_map_rgb = np.reshape(normal_map_rgb, [256,256,3]);
        normal_map_rgb = (((normal_map_rgb + 1) / 2) * 255).astype(np.uint8);
        normal_map_rgb = Image.fromarray(normal_map_rgb)
        normal_map_rgb.save(save_normal_path+data_name+"_normal.png")
        cc+=1
        if nn!=100.0:
            print(data_name,'替换：',nn,'  to  ',loss)
            print('min,max,mean,max-min,max-mean,mean-min',min_depth,max_depth,mean_depth,max_depth-min_depth,max_depth-mean_depth,mean_depth-min_depth)
    return loss

gt_depth_path='/data/liqianlin/Evalation_Hdnet/new_test_tang_1310/depth_txt/'
hunhe=True
# for f in range(1,1271,5):
for f in tang_mat:
    if f[0].startswith('a_'):
        data_name = f[0].split('.')[0]
        data_name1 = f[1].split('.')[0]
        data_name2 = f[2].split('.')[0]
        data_name3 = f[3].split('.')[0]
        data_name4 = f[4].split('_')[1]+'_'+f[4].split('_')[2].split('.')[0]
    else:
        data_name = f[0].split('_')[1]+'_'+f[0].split('_')[2].split('.')[0]
        data_name1 = f[1].split('_')[1]+'_'+f[1].split('_')[2].split('.')[0]
        data_name2 = f[2].split('_')[1]+'_'+f[2].split('_')[2].split('.')[0]
        data_name3 = f[3].split('_')[1]+'_'+f[3].split('_')[2].split('.')[0]
        data_name4 = f[4].split('_')[1]+'_'+f[4].split('_')[2].split('.')[0]

    print('')
    print(int(m/5),'   ',data_name,data_name1,data_name2,data_name3,data_name4)
    # print(3/0)
    # print(f+1,'  Processing file: ',data_name)
    X,Z, Z3,ZB = read_test_data_6_channals_tang(tang,data_name,IMAGE_HEIGHT,IMAGE_WIDTH)
    X1,Z1, Z31,ZB1 = read_test_data_6_channals_tang(tang,data_name1,IMAGE_HEIGHT,IMAGE_WIDTH)
    X2,Z2, Z32, ZB2 = read_test_data_6_channals_tang(tang,data_name2,IMAGE_HEIGHT,IMAGE_WIDTH)
    X3,Z333, Z33, ZB3 = read_test_data_6_channals_tang(tang,data_name3,IMAGE_HEIGHT,IMAGE_WIDTH)
    X4,Z4, Z34, ZB4 = read_test_data_6_channals_tang(tang,data_name4,IMAGE_HEIGHT,IMAGE_WIDTH)

    prediction1n = sess2.run([out2_normal],feed_dict={x1_n:X})
    prediction2n = sess2.run([out2_normal],feed_dict={x1_n:X1})
    prediction3n = sess2.run([out2_normal],feed_dict={x1_n:X2})
    prediction4n = sess2.run([out2_normal],feed_dict={x1_n:X3})
    prediction5n = sess2.run([out2_normal],feed_dict={x1_n:X4})
    
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
    # X_1[...,6]=DP[...,0]
    # X_1[...,7]=DP[...,1]
    # X_1[...,8]=DP[...,2]

    normal_pred_raw1  = np.asarray(prediction2n)[0,...]
    normal_pred1 = nmap_normalization(normal_pred_raw1)
    normal_pred1 = np.where(Z31,normal_pred1,np.zeros_like(normal_pred1))
    X_12 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,6),dtype='f') 
    X_12[...,0]=X1[...,0]
    X_12[...,1]=X1[...,1]
    X_12[...,2]=X1[...,2]
    X_12[...,3]=normal_pred1[...,0]
    X_12[...,4]=normal_pred1[...,1]
    X_12[...,5]=normal_pred1[...,2] 

    normal_pred_raw2  = np.asarray(prediction3n)[0,...]
    normal_pred2 = nmap_normalization(normal_pred_raw2)
    normal_pred2 = np.where(Z32,normal_pred2,np.zeros_like(normal_pred2))
    X_13 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,6),dtype='f') 
    X_13[...,0]=X2[...,0]
    X_13[...,1]=X2[...,1]
    X_13[...,2]=X2[...,2]
    X_13[...,3]=normal_pred2[...,0]
    X_13[...,4]=normal_pred2[...,1]
    X_13[...,5]=normal_pred2[...,2]  

    normal_pred_raw3  = np.asarray(prediction4n)[0,...]
    normal_pred3 = nmap_normalization(normal_pred_raw3)
    normal_pred3 = np.where(Z33,normal_pred3,np.zeros_like(normal_pred3))
    X_14 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,6),dtype='f') 
    X_14[...,0]=X3[...,0]
    X_14[...,1]=X3[...,1]
    X_14[...,2]=X3[...,2]
    X_14[...,3]=normal_pred3[...,0]
    X_14[...,4]=normal_pred3[...,1]
    X_14[...,5]=normal_pred3[...,2]   

    normal_pred_raw4  = np.asarray(prediction5n)[0,...]
    normal_pred4 = nmap_normalization(normal_pred_raw4)
    normal_pred4 = np.where(Z34,normal_pred4,np.zeros_like(normal_pred4))
    X_15 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,6),dtype='f') 
    X_15[...,0]=X4[...,0]
    X_15[...,1]=X4[...,1]
    X_15[...,2]=X4[...,2]
    X_15[...,3]=normal_pred4[...,0]
    X_15[...,4]=normal_pred4[...,1]
    X_15[...,5]=normal_pred4[...,2]
    
    # prediction1 = sess.run([out2_1],feed_dict={x1:X_1})
    # prediction2= sess.run([out2_1],feed_dict={x1:X_12})
    # prediction3 = sess.run([out2_1],feed_dict={x1:X_13})
    # prediction4= sess.run([out2_1],feed_dict={x1:X_14})
    # prediction5 = sess.run([out2_1],feed_dict={x1:X_15})
    prediction11,prediction12,prediction13,prediction14,prediction5 = sess.run([out2_1,out2_2,out2_3,out2_4,out2_5],feed_dict={x1:X_1,x2:X_12,x3:X_13,x4:X_14,x5:X_15})
    prediction12,prediction22,prediction52,prediction32,prediction4 = sess.run([out2_1,out2_2,out2_3,out2_4,out2_5],feed_dict={x1:X_1,x2:X_12,x3:X_15,x4:X_13,x5:X_14})
    prediction13,prediction53,prediction43,prediction23,prediction3 = sess.run([out2_1,out2_2,out2_3,out2_4,out2_5],feed_dict={x1:X_1,x2:X_15,x3:X_14,x4:X_12,x5:X_13})
    prediction54,prediction44,prediction34,prediction14,prediction2 = sess.run([out2_1,out2_2,out2_3,out2_4,out2_5],feed_dict={x1:X_15,x2:X_14,x3:X_13,x4:X_1,x5:X_12})
    prediction55,prediction45,prediction35,prediction25,prediction1 = sess.run([out2_1,out2_2,out2_3,out2_4,out2_5],feed_dict={x1:X_15,x2:X_14,x3:X_13,x4:X_12,x5:X_1})
    prediction6 = sess.run([out2_11],feed_dict={x1:X_1})
    prediction7 = sess.run([out2_11],feed_dict={x1:X_12})
    prediction8 = sess.run([out2_11],feed_dict={x1:X_13})
    prediction9 = sess.run([out2_11],feed_dict={x1:X_14})
    prediction10 = sess.run([out2_11],feed_dict={x1:X_15})

    image  = np.asarray(prediction1)[0,...,0] #prediction1.shape=(1,1,256,256,1)
    imagen = normal_pred[0,...]
    image2  = np.asarray(prediction2)[0,...,0]
    imagen2 = normal_pred1[0,...]    
    image3  = np.asarray(prediction3)[0,...,0]
    imagen3 = normal_pred2[0,...]    
    image4  = np.asarray(prediction4)[0,...,0]
    imagen4 = normal_pred3[0,...]    
    image5  = np.asarray(prediction5)[0,...,0]
    imagen5 = normal_pred4[0,...]

    image6  = np.asarray(prediction6)[0,0,...,0] #prediction1.shape=(1,1,256,256,1)
    image7  = np.asarray(prediction7)[0,0,...,0]
    image8  = np.asarray(prediction8)[0,0,...,0]
    image9  = np.asarray(prediction9)[0,0,...,0]
    image10  = np.asarray(prediction10)[0,0,...,0]

    image61  = np.asarray(prediction11)[0,...,0] 
    image62  = np.asarray(prediction12)[0,...,0]
    image63  = np.asarray(prediction13)[0,...,0]
    image64  = np.asarray(prediction14)[0,...,0]

    image71  = np.asarray(prediction12)[0,...,0] 
    image72  = np.asarray(prediction22)[0,...,0]
    image73  = np.asarray(prediction23)[0,...,0]
    image74  = np.asarray(prediction25)[0,...,0]

    image81  = np.asarray(prediction13)[0,...,0] 
    image82  = np.asarray(prediction32)[0,...,0]
    image83  = np.asarray(prediction34)[0,...,0]
    image84  = np.asarray(prediction35)[0,...,0]

    image91  = np.asarray(prediction14)[0,...,0] 
    image92  = np.asarray(prediction43)[0,...,0]
    image93  = np.asarray(prediction44)[0,...,0]
    image94  = np.asarray(prediction45)[0,...,0]

    image101  = np.asarray(prediction52)[0,...,0] 
    image102  = np.asarray(prediction53)[0,...,0]
    image103  = np.asarray(prediction54)[0,...,0]
    image104  = np.asarray(prediction55)[0,...,0]

    if hunhe:
        if data_name.startswith('a_'):
            loss10 = comp_deptherr(ZB4,gt_depth_path,data_name4,image10,Z4,Z34,imagen5)
            loss101 = comp_deptherr(ZB4,gt_depth_path,data_name4,image101,Z4,Z34,imagen5)
            loss102 = comp_deptherr(ZB4,gt_depth_path,data_name4,image102,Z4,Z34,imagen5)
            loss103 = comp_deptherr(ZB4,gt_depth_path,data_name4,image103,Z4,Z34,imagen5)
            loss104 = comp_deptherr(ZB4,gt_depth_path,data_name4,image104,Z4,Z34,imagen5)
            loss5 = comp_deptherr(ZB4,gt_depth_path,data_name4,image5,Z4,Z34,imagen5)
        else:
            loss6 = comp_deptherr(ZB,gt_depth_path,data_name,image6,Z,Z3,imagen)
            loss61 = comp_deptherr(ZB,gt_depth_path,data_name,image61,Z,Z3,imagen)
            loss62 = comp_deptherr(ZB,gt_depth_path,data_name,image62,Z,Z3,imagen)
            loss63 = comp_deptherr(ZB,gt_depth_path,data_name,image63,Z,Z3,imagen)
            loss64 = comp_deptherr(ZB,gt_depth_path,data_name,image64,Z,Z3,imagen)

            loss7 = comp_deptherr(ZB1,gt_depth_path,data_name1,image7,Z1,Z31,imagen2)
            loss71 = comp_deptherr(ZB1,gt_depth_path,data_name1,image71,Z1,Z31,imagen2)
            loss72 = comp_deptherr(ZB1,gt_depth_path,data_name1,image72,Z1,Z31,imagen2)
            loss73 = comp_deptherr(ZB1,gt_depth_path,data_name1,image73,Z1,Z31,imagen2)
            loss74 = comp_deptherr(ZB1,gt_depth_path,data_name1,image74,Z1,Z31,imagen2)    

            loss8 = comp_deptherr(ZB2,gt_depth_path,data_name2,image8,Z2,Z32,imagen3)
            loss81 = comp_deptherr(ZB2,gt_depth_path,data_name2,image81,Z2,Z32,imagen3)
            loss82 = comp_deptherr(ZB2,gt_depth_path,data_name2,image82,Z2,Z32,imagen3)
            loss83 = comp_deptherr(ZB2,gt_depth_path,data_name2,image83,Z2,Z32,imagen3)
            loss84 = comp_deptherr(ZB2,gt_depth_path,data_name2,image84,Z2,Z32,imagen3)

            loss9 = comp_deptherr(ZB3,gt_depth_path,data_name3,image9,Z333,Z33,imagen4)
            loss91 = comp_deptherr(ZB3,gt_depth_path,data_name3,image91,Z333,Z33,imagen4)
            loss92 = comp_deptherr(ZB3,gt_depth_path,data_name3,image92,Z333,Z33,imagen4)
            loss93 = comp_deptherr(ZB3,gt_depth_path,data_name3,image93,Z333,Z33,imagen4)
            loss94 = comp_deptherr(ZB3,gt_depth_path,data_name3,image94,Z333,Z33,imagen4)

            loss10 = comp_deptherr(ZB4,gt_depth_path,data_name4,image10,Z4,Z34,imagen5)
            loss101 = comp_deptherr(ZB4,gt_depth_path,data_name4,image101,Z4,Z34,imagen5)
            loss102 = comp_deptherr(ZB4,gt_depth_path,data_name4,image102,Z4,Z34,imagen5)
            loss103 = comp_deptherr(ZB4,gt_depth_path,data_name4,image103,Z4,Z34,imagen5)
            loss104 = comp_deptherr(ZB4,gt_depth_path,data_name4,image104,Z4,Z34,imagen5)

            loss1 = comp_deptherr(ZB,gt_depth_path,data_name,image,Z,Z3,imagen)
            loss2 = comp_deptherr(ZB1,gt_depth_path,data_name1,image2,Z1,Z31,imagen2)
            loss3 = comp_deptherr(ZB2,gt_depth_path,data_name2,image3,Z2,Z32,imagen3)
            loss4 = comp_deptherr(ZB3,gt_depth_path,data_name3,image4,Z333,Z33,imagen4)
            loss5 = comp_deptherr(ZB4,gt_depth_path,data_name4,image5,Z4,Z34,imagen5)
    else:
        if data_name.startswith('a_'):
            loss5 = comp_deptherr(ZB4,gt_depth_path,data_name4,image5,Z4,Z34,imagen5) 
        else:
            loss1 = comp_deptherr(ZB,gt_depth_path,data_name,image,Z,Z3,imagen)
            loss2 = comp_deptherr(ZB1,gt_depth_path,data_name1,image2,Z1,Z31,imagen2)
            loss3 = comp_deptherr(ZB2,gt_depth_path,data_name2,image3,Z2,Z32,imagen3)
            loss4 = comp_deptherr(ZB3,gt_depth_path,data_name3,image4,Z333,Z33,imagen4)
            loss5 = comp_deptherr(ZB4,gt_depth_path,data_name4,image5,Z4,Z34,imagen5)
    
    if visualization:
        depth_map = image*Z[0,...,0]
        normal_map = imagen*Z3[0,...]
        min_depth = np.amin(depth_map[depth_map>0])
        max_depth = np.amax(depth_map[depth_map>0])
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth

        depth_map11 = image6*Z[0,...,0]
        min_depth = np.amin(depth_map11[depth_map11>0])
        max_depth = np.amax(depth_map11[depth_map11>0])
        depth_map11[depth_map11 < min_depth] = min_depth
        depth_map11[depth_map11 > max_depth] = max_depth

        normal_map_rgb = -1*normal_map
        normal_map_rgb[...,2] = -1*((normal_map[...,2]*2)+1)
        normal_map_rgb = np.reshape(normal_map_rgb, [256,256,3]);
        normal_map_rgb = (((normal_map_rgb + 1) / 2) * 255).astype(np.uint8);
        
        if data_name.startswith('a_'):
            depth_map4 = image5*Z4[0,...,0]
            normal_map4 = imagen5*Z34[0,...]
            min_depth4 = np.amin(depth_map4[depth_map4>0])
            max_depth4 = np.amax(depth_map4[depth_map4>0])
            depth_map4[depth_map4 < min_depth4] = min_depth4
            depth_map4[depth_map4 > max_depth4] = max_depth4
            
            depth_map55 = image10*Z4[0,...,0]
            min_depth4 = np.amin(depth_map55[depth_map55>0])
            max_depth4 = np.amax(depth_map55[depth_map55>0])
            depth_map55[depth_map55 < min_depth4] = min_depth4
            depth_map55[depth_map55 > max_depth4] = max_depth4

            normal_map_rgb4 = -1*normal_map4
            normal_map_rgb4[...,2] = -1*((normal_map4[...,2]*2)+1)
            normal_map_rgb4 = np.reshape(normal_map_rgb4, [256,256,3]);
            normal_map_rgb4 = (((normal_map_rgb4 + 1) / 2) * 255).astype(np.uint8);
            
            plt.imsave(save_depth_path+data_name4+"_depth.png", depth_map4, cmap="hot") 
            d = np.array(scipy.misc.imread(save_depth_path+data_name4+"_depth.png"),dtype='f')
            d = np.where(Z3[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name4+"_depth.png")
            plt.imsave(save_depth_path+data_name4+"_depth_r.png", depth_map55, cmap="hot") 
            d = np.array(scipy.misc.imread(save_depth_path+data_name4+"_depth_r.png"),dtype='f')
            d = np.where(Z3[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name4+"_depth_r.png")
            plt.imsave(save_depth_path+data_name+"_depth.png", depth_map, cmap="hot") 
            d = np.array(scipy.misc.imread(save_depth_path+data_name+"_depth.png"),dtype='f')
            d = np.where(Z3[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name+"_depth.png")
            plt.imsave(save_depth_path+data_name+"_depth_r.png", depth_map11, cmap="hot") 
            d = np.array(scipy.misc.imread(save_depth_path+data_name+"_depth_r.png"),dtype='f')
            d = np.where(Z3[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name+"_depth_r.png")
        else:
            normal_map_rgb = Image.fromarray(normal_map_rgb)
            normal_map_rgb.save(save_normal_path+data_name+"_normal.png")
            d = np.array(scipy.misc.imread(save_depth_path+data_name+"_depth.png"),dtype='f')
            d = np.where(Z3[0,...]>0,d[...,0:3],255.0)
            n = np.array(scipy.misc.imread(save_normal_path+data_name+"_normal.png"),dtype='f')
            n = np.where(Z3[0,...]>0,n[...,0:3],255.0)
            final_im = get_concat_h(Image.fromarray(np.uint8(X[0,...])),Image.fromarray(np.uint8(d)))
            final_im = get_concat_h(final_im,Image.fromarray(np.uint8(n)))
            final_im.save(Vis_dir+data_name+"_results.png")

            depth_map1 = image2*Z1[0,...,0]
            normal_map1 = imagen2*Z31[0,...]
            min_depth1 = np.amin(depth_map1[depth_map1>0])
            max_depth1 = np.amax(depth_map1[depth_map1>0])
            depth_map1[depth_map1 < min_depth1] = min_depth1
            depth_map1[depth_map1 > max_depth1] = max_depth1
            
            depth_map22 = image7*Z1[0,...,0]
            min_depth1 = np.amin(depth_map22[depth_map22>0])
            max_depth1 = np.amax(depth_map22[depth_map22>0])
            depth_map22[depth_map22 < min_depth1] = min_depth1
            depth_map22[depth_map22 > max_depth1] = max_depth1
        
            plt.imsave(save_depth_path+data_name1+"_depth.png", depth_map1, cmap="hot")
            d = np.array(scipy.misc.imread(save_depth_path+data_name1+"_depth.png"),dtype='f')
            d = np.where(Z31[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name1+"_depth.png") 
            plt.imsave(save_depth_path+data_name1+"_depth_r.png", depth_map22, cmap="hot")
            d = np.array(scipy.misc.imread(save_depth_path+data_name1+"_depth_r.png"),dtype='f')
            d = np.where(Z31[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name1+"_depth_r.png") 
            
            depth_map2 = image3*Z2[0,...,0]
            normal_map2 = imagen3*Z32[0,...]
            min_depth2 = np.amin(depth_map2[depth_map2>0])
            max_depth2 = np.amax(depth_map2[depth_map2>0])
            depth_map2[depth_map2 < min_depth2] = min_depth2
            depth_map2[depth_map2 > max_depth2] = max_depth2
            
            depth_map33 = image8*Z2[0,...,0]
            min_depth2 = np.amin(depth_map33[depth_map33>0])
            max_depth2 = np.amax(depth_map33[depth_map33>0])
            depth_map33[depth_map33 < min_depth2] = min_depth2
            depth_map33[depth_map33 > max_depth2] = max_depth2
            
            plt.imsave(save_depth_path+data_name2+"_depth.png", depth_map2, cmap="hot") 
            d = np.array(scipy.misc.imread(save_depth_path+data_name2+"_depth.png"),dtype='f')
            d = np.where(Z32[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name2+"_depth.png")
            plt.imsave(save_depth_path+data_name2+"_depth_r.png", depth_map33, cmap="hot") 
            d = np.array(scipy.misc.imread(save_depth_path+data_name2+"_depth_r.png"),dtype='f')
            d = np.where(Z32[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name2+"_depth_r.png")
    
            depth_map3 = image4*Z333[0,...,0]
            normal_map3 = imagen4*Z33[0,...]
            min_depth3 = np.amin(depth_map3[depth_map3>0])
            max_depth3 = np.amax(depth_map3[depth_map3>0])
            depth_map3[depth_map3 < min_depth3] = min_depth3
            depth_map3[depth_map3 > max_depth3] = max_depth3
            
            depth_map44 = image9*Z333[0,...,0]
            min_depth3 = np.amin(depth_map44[depth_map44>0])
            max_depth3 = np.amax(depth_map44[depth_map44>0])
            depth_map44[depth_map44 < min_depth3] = min_depth3
            depth_map44[depth_map44 > max_depth3] = max_depth3

            normal_map_rgb3 = -1*normal_map3
            normal_map_rgb3[...,2] = -1*((normal_map3[...,2]*2)+1)
            normal_map_rgb3 = np.reshape(normal_map_rgb3, [256,256,3]);
            normal_map_rgb3 = (((normal_map_rgb3 + 1) / 2) * 255).astype(np.uint8);
            
            plt.imsave(save_depth_path+data_name3+"_depth.png", depth_map3, cmap="hot") 
            d = np.array(scipy.misc.imread(save_depth_path+data_name3+"_depth.png"),dtype='f')
            d = np.where(Z33[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name3+"_depth.png")
            plt.imsave(save_depth_path+data_name3+"_depth_r.png", depth_map44, cmap="hot") 
            d = np.array(scipy.misc.imread(save_depth_path+data_name3+"_depth_r.png"),dtype='f')
            d = np.where(Z33[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name3+"_depth_r.png")
            
            depth_map4 = image5*Z4[0,...,0]
            normal_map4 = imagen5*Z34[0,...]
            min_depth4 = np.amin(depth_map4[depth_map4>0])
            max_depth4 = np.amax(depth_map4[depth_map4>0])
            depth_map4[depth_map4 < min_depth4] = min_depth4
            depth_map4[depth_map4 > max_depth4] = max_depth4
            
            depth_map55 = image10*Z4[0,...,0]
            min_depth4 = np.amin(depth_map55[depth_map55>0])
            max_depth4 = np.amax(depth_map55[depth_map55>0])
            depth_map55[depth_map55 < min_depth4] = min_depth4
            depth_map55[depth_map55 > max_depth4] = max_depth4

            normal_map_rgb4 = -1*normal_map4
            normal_map_rgb4[...,2] = -1*((normal_map4[...,2]*2)+1)
            normal_map_rgb4 = np.reshape(normal_map_rgb4, [256,256,3]);
            normal_map_rgb4 = (((normal_map_rgb4 + 1) / 2) * 255).astype(np.uint8);
            
            plt.imsave(save_depth_path+data_name4+"_depth.png", depth_map4, cmap="hot") 
            d = np.array(scipy.misc.imread(save_depth_path+data_name4+"_depth.png"),dtype='f')
            d = np.where(Z3[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name4+"_depth.png")
            plt.imsave(save_depth_path+data_name4+"_depth_r.png", depth_map55, cmap="hot") 
            d = np.array(scipy.misc.imread(save_depth_path+data_name4+"_depth_r.png"),dtype='f')
            d = np.where(Z3[0,...]>0,d[...,0:3],255.0);d=Image.fromarray(np.uint8(d))
            d.save(save_depth_path_white+data_name4+"_depth_r.png")
    m+=5
print(sec_path,'总的替换数量：',cc)