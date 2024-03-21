import pandas as pd
import numpy as np
import math
import tensorflow as tf
import skimage.data
from PIL import Image, ImageDraw, ImageFont
from tensorflow.python.platform import gfile
import scipy.misc
import matplotlib.pyplot as plt
import os.path
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

name_csv_path="/home/laihuaijing/liqianlin/HDNet_TikTok/training/training_data/surreal_data/surreal_y_x_train_name.csv"
train_point_csv_path="/home/laihuaijing/liqianlin/HDNet_TikTok/training/training_data/surreal_data/new_surreal_y_x_train_to256.csv"
depth_path="/home/laihuaijing/liqianlin/HDNet_TikTok/training/training_data/surreal_data/depth/"
#1048425
mask_path="/home/laihuaijing/liqianlin/HDNet_TikTok/training/training_data/surreal_data/mask/"
def read_csv(name_csv_path,train_point_csv_path):
    temp=1   #surreal_y_x_train.csv中第一行时列名
    csv_file=pd.read_csv(name_csv_path,header=None,usecols=[0,2])
    csv_file_arr=csv_file.values.tolist()
    all_num_image=len(csv_file_arr)  # 3863
    # print(all_num_image)
    #point_row记录点对的累加，以便快速定位该图片所有点对的行；
    #point_row_single_pic记录每张照片的关节点对个数，最大为276；
    point_row=np.zeros(all_num_image)
    point_row_single_pic=np.zeros(all_num_image)
    for i in range(all_num_image):
        temp1=int(csv_file_arr[i][1])
        temp=temp+temp1
        # print(temp)
        point_row_single_pic[i]=temp1
        point_row[i]=temp  #point_row=[276,276+276,...]
        # print(point_row[i])
    train_point_csv=pd.read_csv(train_point_csv_path,header=None,usecols=[0,1,2,3,4])
    train_point_csv_arr=train_point_csv.values.tolist()
    # print(type(train_point_csv_arr))
    
    print("读csv完成")
    return csv_file_arr,train_point_csv_arr,all_num_image,point_row

def read_morerow_csv(name_csv_path,train_point_csv_path):
    #训练的csv包括29*3610=104690张图片，分为145列，999970行
    temp=1   #surreal_y_x_train.csv中第一行时列名
    csv_file=pd.read_csv(name_csv_path,header=None,usecols=[0,1])  # 第一列是 序号 ，第二列是 路径加图片名称
    csv_file_arr=csv_file.values.tolist()
    random.shuffle(csv_file_arr)
    # print(csv_file_arr[0])
    # print(3/0)
    all_num_image=len(csv_file_arr)  # 104690
    train_point_csv_arr=[]
    j=0
    while j<=140:
        train_point_csv=pd.read_csv(train_point_csv_path,header=None,usecols=[j,j+1,j+2,j+3,j+4])
        train_point_csv_arr1=train_point_csv.values.tolist()
        train_point_csv_arr.append(train_point_csv_arr1)
        j+=5
    print("读csv完成")
    # print(len(train_point_csv_arr),train_point_csv_arr[2][999968]) #len=28999130
    # print(3/0)
    return csv_file_arr,train_point_csv_arr,all_num_image
def get_more_order_patch(depth_path,train_point_csv_arr,Batch_size,itr,csv_file_arr):
    #读取csv文件,获取每一行的第一列 文件地址 和第三列 点的个数(usecols)
    point=[]  
    name=[]
    batch_color=[]
    batch_mask=[]
    name_chose=csv_file_arr[itr]
    color_name=name_chose[1]
    name= (color_name.split("/"))[9]
    print(name)
    # print(3/0)
    depth_name=(((color_name.split("/"))[9]).split("."))[0]+'_depth.txt'
    batch_color.append(np.array(Image.open(color_name)))
    batch_mask.append(np.loadtxt(depth_path + depth_name))
    batch_mask = np.array(batch_mask,dtype='f')
    batch_color = np.array(batch_color,dtype='f')

    x1_surr = np.zeros((Batch_size,256,256,3),dtype='f')
    x1_surr[0,...] = batch_color
    mask = np.zeros((Batch_size,256,256,1),dtype='b')
    mask[0,...,0] = batch_mask > 0
    Z1_surr = np.zeros((Batch_size,256,256,3),dtype='b')
    Z1_surr[...,0]=mask[...,0]
    Z1_surr[...,1]=mask[...,0]
    Z1_surr[...,2]=mask[...,0]
    # X1为白色背景图的color
    x1_surr = np.where(Z1_surr,x1_surr,np.ones_like(x1_surr)*255.0)

    point=handle_morecsv_point(train_point_csv_arr,name_chose)

    return x1_surr,name,point,mask

def handle_morecsv_point(train_point_csv_arr,name_chose):
    #根据随机选取的文件以及其对应的点对个数
    #定位到所选图片所在surreal_y_x_train.csv中点对的行数，以及点对数
    file_num=int(name_chose[0])   # 定位第几张图片
    list_num=int(file_num/3610)        # 得到的商既可以定位到在那个list中，总共有29个list
    point_num=int((file_num%3610)*277) # 得到的余数可以定位到数据的行数，即带有路径的行数
    # file_strar=train_point_csv_arr[list_num][]
    # print(name_chose[1],train_point_csv_arr[list_num][point_num][0])
    if name_chose[1]==train_point_csv_arr[list_num][point_num][0]:
        file_point=train_point_csv_arr[list_num][point_num+1:point_num+277]
    else:
        # print(file_point)
        print(name_chose[1],file_num,list_num,point_num) 
        print('name.csv train.csv no match')
        print(3/0)  
    # print(3/0)
    return file_point

def get_order_patch(mask_path,train_point_csv_arr,Batch_size,all_num_image,csv_file_arr,point_row):
    #读取csv文件,获取每一行的第一列 文件地址 和第三列 点的个数(usecols)
    name_list=[]
    name_list_row=[]
    point=[]  
    name=[]
    batch_color=[]
    batch_mask=[]
    csv_random_Bsize=np.random.choice(all_num_image,Batch_size).tolist()
    for i in range(len(csv_random_Bsize)):
        name_list_row.append(csv_random_Bsize[i])
        name_list.append(csv_file_arr[csv_random_Bsize[i]])
        # print(name_list)
        color_name=csv_file_arr[csv_random_Bsize[i]][0]
        mask_name=(((color_name.split("/"))[9]).split("."))[0]+'_mask.png'
        name.append(csv_file_arr[csv_random_Bsize[i]][0])
        batch_color.append(np.array(Image.open(color_name)))
        batch_mask.append(np.array(Image.open(mask_path + mask_name)))
    # print(len(name_list))
    # print(name_list,name_list_row)
    # print(name)
    point=handle_csv_point(name_list,name_list_row,train_point_csv_arr,point_row)
    batch_mask = np.array(batch_mask,dtype='f')
    batch_color = np.array(batch_color,dtype='f')

    x1_surr = np.zeros((Batch_size,256,256,3),dtype='f')
    x1_surr = batch_color

    mask = np.zeros((Batch_size,256,256,1),dtype='b')
    mask[...,0] = batch_mask > 0
    Z1_surr = np.zeros((Batch_size,256,256,3),dtype='b')
    Z1_surr[...,0]=mask[...,0]
    Z1_surr[...,1]=mask[...,0]
    Z1_surr[...,2]=mask[...,0]
    # X1为白色背景图的color
    x1_surr = np.where(Z1_surr,x1_surr,np.ones_like(x1_surr)*255.0)
    return x1_surr,name,name_list,name_list_row,point,mask

def get_surreal_patch(mask_path,train_point_csv_arr,surr_num,all_num_image,csv_file_arr,point_row):
    #读取csv文件,获取每一行的第一列 文件地址 和第三列 点的个数(usecols)
    name_list=[]
    name_list_row=[]
    point=[]  
    name=[]
    batch_color=[]
    batch_depth=[]
    batch_mask=[]
    csv_random_Bsize=np.random.choice(all_num_image,surr_num).tolist()
    for i in range(len(csv_random_Bsize)):
        name_list_row.append(csv_random_Bsize[i])
        name_list.append(csv_file_arr[csv_random_Bsize[i]])
        # print(name_list)
        color_name=csv_file_arr[csv_random_Bsize[i]][0]
        depth_name=(((color_name.split("/"))[9]).split("."))[0]+'_depth.txt'
        mask_name=(((color_name.split("/"))[9]).split("."))[0]+'_mask.png'
        name.append(csv_file_arr[csv_random_Bsize[i]][0])
        batch_color.append(np.array(Image.open(color_name)))
        batch_depth.append(np.genfromtxt(depth_path + depth_name))
        batch_mask.append(np.array(Image.open(mask_path + mask_name)))
    print(len(name_list))
    # print(name_list,name_list_row)
    # print(name)
    point=handle_csv_point(name_list,name_list_row,train_point_csv_arr,point_row)
    batch_mask = np.array(batch_mask,dtype='f')
    batch_color = np.array(batch_color,dtype='f')
    batch_depth = np.array(batch_depth,dtype='f')
    x1_surr = np.zeros((surr_num,256,256,3),dtype='f')
    x1_surr = batch_color
    x2_ =np.zeros((surr_num,256,256,3),dtype='f')
    x2_= batch_color
    batch_depth = np.array(batch_depth,dtype='f')
    Y1 = np.zeros((surr_num,256,256,1),dtype='f')
    Y1[...,0] = batch_depth
    mask = np.zeros((surr_num,256,256,1),dtype='b')
    mask[...,0] = batch_mask > 0
    Z1_surr = np.zeros((surr_num,256,256,3),dtype='b')
    Z1_surr[...,0]=mask[...,0]
    Z1_surr[...,1]=mask[...,0]
    Z1_surr[...,2]=mask[...,0]
    # X1为白色背景图的color
    Y1_surr = np.where(mask,Y1,np.zeros_like(Y1))
    x1_surr = np.where(Z1_surr,x1_surr,np.ones_like(x1_surr)*255.0)
    x2_ = np.where(Z1_surr,x2_,np.ones_like(x2_)*255.0)
    x2_surr =np.zeros((surr_num,256,256,9),dtype='f')
    temp=np.zeros((surr_num,256,256,6),dtype='f')
    x2_surr[...,0]=x2_[...,0]
    x2_surr[...,1]=x2_[...,1]
    x2_surr[...,2]=x2_[...,2]
    x2_surr[...,3]=temp[...,0]
    x2_surr[...,4]=temp[...,1]
    x2_surr[...,5]=temp[...,2]
    x2_surr[...,6]=temp[...,3]
    x2_surr[...,7]=temp[...,4]
    x2_surr[...,8]=temp[...,5]
    #判断depth读取某个特定的坐标的深度是按照depth[y,x],还是[x,y];前提是csv中的数据是按照[y,x]顺序存储的；
    #从isnot_order函数得出，depth[y,x]是正确的，比如csv中前五项[160 145 95 154],depth1[159][144],depth[94][153]
    # isnot_order(point,name,depth_path)
    return x2_surr,x1_surr,name,name_list,name_list_row,point,Y1_surr,mask

# name_list: 如 [['/home/laihuaijing/liqianlin/surreal_relation/dataset_test_run0_surreal/some_run0_rgb/30_20_c0002_26.png', 276], ['/home/laihuaijing/liqianlin/surreal_relation/dataset_test_run0_surreal/some_run0_rgb/03_03_c0021_6.png', 276]] [2610, 678]
# name_list_row: 如 [2314, 3372] （surreal_y_x_train_name.csv中选中的图片行数）

def handle_csv_point(name_list,name_list_row,train_point_csv_arr,point_row):
    #根据随机选取的文件以及其对应的点对个数
    point=[]
    # point_arr_number=1
    point_arr_number=len(name_list_row)
    for i in range(point_arr_number):
        #定位到所选图片所在surreal_y_x_train.csv中点对的行数，以及点对数
        file_num=int(name_list_row[i])#定位第几张图片
        if file_num==0:
            file_strar=int(name_list_row[i]+1+2)#定位该张图片的第一行这一行坐标以及order
            file_end=int(name_list_row[i]+point_row[name_list_row[i]]+1)
            file_point=train_point_csv_arr[file_strar-1:file_end]
            # print(point)
        else:
            file_strar=int(name_list_row[i]+point_row[name_list_row[i]-1]+2)#定位该张图片的第一行这一行坐标以及order
            file_end=int(name_list_row[i]+point_row[name_list_row[i]]+1)
            # print(i,name_list_row[i])#定位该张图片的最后一行这一行坐标以及order
            # print(name_list_row[i]+point_row[name_list_row[i]-1]+2)
            # print(name_list_row[i]+point_row[name_list_row[i]]+1)
            # print(type(train_point_csv_arr))
            file_point=train_point_csv_arr[file_strar-1:file_end]
        # print(file_point)
        point.append(file_point)
    # print(point)
    # print(len(point))
    # print(point.shape)
    # print("-------------------------")
    # temp=point[0]
    # temp=np.array(temp)
    # print(temp)
    # print(len(temp))
    # print(temp.shape)
    # print(3/0)
    # 返回的point是一个列表，列表没有shape
    return point
def isnot_order(point,name,depth_path):
    lens=len(name)
    j=0
    for i in range(lens):
        point=np.array(point[i])
        row=point.shape[0]
        name=str(np.array(name[i]))
        print(name)
        print((name.split("/"))[9])
        depth_name=(((name.split("/"))[9]).split("."))[0]+'_depth.txt'
        print(depth_path+depth_name)
        depth=np.loadtxt(depth_path+depth_name)
        print(depth.shape)
        for j in range(row):
            z_A_x=int(point[j][0]);z_A_y=int(point[j][1])
            z_B_x=int(point[j][2]);z_B_y=int(point[j][3])
            ord=point[j][4]
            if z_A_x>256 or z_A_x<0 or z_A_y<0 or z_A_y>256 or z_B_y>256 or  z_B_y<0 or  z_B_x<0 or z_B_x>256:
                continue
            #坐标-1，因为csv中的坐标是从1开始的从而得到order,但是python坐标是从0开始的；
            d1=depth[z_A_x-1][z_A_y-1]
            d2=depth[z_B_x-1][z_B_y-1]
            ord1=d1-d2
            if ord1==0:
                ord1="="
            else:
                if ord1>0:
                    ord1=">"
                else:
                    ord1="<"
            if (ord!=ord1):
                print(z_A_x,z_A_y,z_B_x,z_B_y,d1,d2,ord,ord1)
            else:
                j+=1
                print(row,j)
            print("---------")


# surreal*****************************************************************************************************
def calc_order(output,y):
    # y是point中指定的列表，如point[298],化成数组后shape(253,5),output是四维数组[1,256,256,1]
    loss_order=tf.constant(0.0)
    #1. 先将表格中的order化为1("<")，-1(">")，0("=")
    #2. rk={1,-1,0},rk代表的是gt的order关系
    #3. rk=1,loss=log{1+exp(-di+dj)}
    #   rk=-1,loss=log{1+exp(di-dj)}
    #   rk=-1,loss=(di-dj)*(di-dj)
  
    for i in range(276):
        # loss_order=tf.constant(3.0)
        A_x=y[i,0]
        A_y=y[i,1]
        B_x=y[i,2]
        B_y=y[i,3]
        ground_truth=y[i,4]
        # if ground_truth=='>':
        #     ground_truth=1
        # else :
        #     if ground_truth=='<':
        #         ground_truth=-1
        #     else:
        #         ground_truth=0
        ground_truth=tf.cast(ground_truth,tf.float32)
        #在csv中虽然截取了depth的中间部分但是有一些原先存在的关节点在截取之外，需要剔除；       
        z_A=output[0,A_x-1,A_y-1,0]
        z_B=output[0,B_x-1,B_y-1,0]
        if ground_truth==0:
            loss_order += tf.math.max(0.0, (z_A - z_B) * (z_A - z_B) )
        else:
            loss_order +=tf.math.log( 1.0 + tf.math.exp( - ground_truth * (z_A - z_B) ) )
    # loss_order=tf.truediv(loss_order, 276.0)
    #判断是否loss_order需要转换为tensor变量
    return loss_order

def calc_tk_joins(keypoint,tk_order,out_tk):
    # tk_order=tk_order[:,:,0]
    # print('x_shape',tk_order.shape)
    loss_order=tf.constant(0.0)
    for i in range(17):
        A_x=keypoint[i,0]
        A_y=keypoint[i,1]
        gt_A=tk_order[A_y,A_x]
        z_A=out_tk[0,A_y,A_x,0]
        # print('gt_A',gt_A)
        for j in range(i+1,17):
            B_x=keypoint[j,0]
            B_y=keypoint[j,1]
            gt_B=tk_order[B_y,B_x]
            z_B=out_tk[0,B_y,B_x,0]
            loss=tf.cond(tf.less(gt_B,gt_A),lambda:tf.math.log( 1.0 + tf.math.exp( -1* (z_A - z_B) ) ),lambda:tf.cond(tf.equal(gt_A,gt_B),lambda:((z_A - z_B) * (z_A - z_B)),lambda:tf.math.log( 1.0 + tf.math.exp( (z_A - z_B) ) )))
            loss_order += loss
    loss_order=tf.truediv(loss_order, 136.0)
    return loss_order



# csv_file_arr,train_point_csv_arr,all_num_image,point_row=read_csv(name_csv_path,train_point_csv_path)
# x1_surr,name,name_list,name_list_row,point=get_surreal_patch(mask_path,train_point_csv_arr,10,all_num_image,csv_file_arr,point_row)