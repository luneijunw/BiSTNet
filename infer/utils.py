import tensorflow as tf
import numpy as np
# import skimage.data
from PIL import Image, ImageDraw, ImageFont
import math
from tensorflow.python.platform import gfile
import scipy.misc
import glob
import ntpath
import os
from os import path
import pandas as pd
import matplotlib.pyplot as plt


IMAGE_HEIGHT=256
IMAGE_WIDTH=256
surr_color_path='/data/liqianlin/Evalation_Hdnet/surreal_order_test_662_256/color_wobg/'
surr_mask_path='/data/liqianlin/Evalation_Hdnet/surreal_order_test_662_256/mask_val/'
# **********************************************************************************************************
def write_matrix_txt(a,filename):
    mat = np.matrix(a)
    with open(filename,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.5f')
            
# **********************************************************************************************************
def get_origin_scaling(bbs, IMAGE_HEIGHT):
    Bsz = np.shape(bbs)[0]
    batch_origin = []
    batch_scaling = []
    
    for i in range(Bsz):
        bb1_t = bbs[i,...] - 1
        bbc1_t = bb1_t[2:4,0:3]
        
        origin = np.multiply([bb1_t[1,0]-bbc1_t[1,0],bb1_t[0,0]-bbc1_t[0,0]],2)

        squareSize = np.maximum(bb1_t[0,1]-bb1_t[0,0]+1,bb1_t[1,1]-bb1_t[1,0]+1);
        scaling = [np.multiply(np.true_divide(squareSize,IMAGE_HEIGHT),2)]
    
        batch_origin.append(origin)
        batch_scaling.append(scaling)
    
    batch_origin = np.array(batch_origin,dtype='f')
    batch_scaling = np.array(batch_scaling,dtype='f')
    
    O = np.zeros((Bsz,1,2),dtype='f')
    O = batch_origin
    
    S = np.zeros((Bsz,1),dtype='f')
    S = batch_scaling
    
    return O, S

# **********************************************************************************************************
def get_surr_test_data(inpath):
    filename_list = []
    filename_list = os.listdir(inpath)
    return filename_list

def read_surr_test_data_9(csv_file_arr,f,IMAGE_HEIGHT,IMAGE_WIDTH):
    data_name,point2=read_csv(csv_file_arr,f)
    data_name=(data_name.split("/"))[8]
    image_path = surr_color_path + data_name
    print('image_path   ',f,image_path)
    # 
    mask_path = surr_mask_path + data_name.split(".")[0]+'_mask.png'
    dp_path = '/data/liqianlin/Evalation_Hdnet/surreal_order_test_662_256/densepose_val/' + data_name.split(".")[0]+'_IUV.png'
    # print(mask_path)
    # print(3/0)
    # image_path = data_main_path +"/" + data_name + "_img.png"
    # mask_path = data_main_path +"/" + data_name + "_mask.png"
    # dp_path = data_main_path +"/" + data_name + "_dp.png"
    
    color = np.array(scipy.misc.imread(image_path),dtype='f')
    mask = np.array(scipy.misc.imread(mask_path),dtype='f')
    dp = np.array(scipy.misc.imread(dp_path),dtype='f')
    
    X = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    X[0,...] = color
    
    Z = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
    Z[0,...,0] = mask>0
    
    DP = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    DP[0,...] = dp
    
    Z2C3 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
    Z2C3[...,0]=Z[...,0]
    Z2C3[...,1]=Z[...,0]
    Z2C3[...,2]=Z[...,0]
    
    X = np.where(Z2C3,X,np.ones_like(X)*255.0)
    
    Z3 = Z2C3
    
    # camera
    C = np.zeros((3,4),dtype='f')
    C[0,0]=1
    C[1,1]=1
    C[2,2]=1
    
    R = np.zeros((3,3),dtype='f')
    R[0,0]=1
    R[1,1]=1
    R[2,2]=1
    
    Rt = R
    
    K = np.zeros((3,3),dtype='f')
    K[0,0]=1111.6
    K[1,1]=1111.6

    K[0,2]=960
    K[1,2]=540
    K[2,2]=1
    
    Ki = np.linalg.inv(K)
    cen = np.zeros((3),dtype='f')
    bbs = np.array([[25,477],[420,872],[1,453],[1,453]],dtype='f')
    bbs = np.reshape(bbs,[1,4,2])
    (origin, scaling) = get_origin_scaling(bbs, IMAGE_HEIGHT)
    
    return X,Z, Z3, C, cen,K,Ki, R,Rt,  scaling, origin,DP,point2,data_name

def read_surr_test_data_6(csv_file_arr,f,IMAGE_HEIGHT,IMAGE_WIDTH):
    data_name,point2=read_csv(csv_file_arr,f)
    data_name=(data_name.split("/"))[8]
    image_path = surr_color_path + data_name
    print('image_path   ',f,image_path)
    # 
    mask_path = surr_mask_path + data_name.split(".")[0]+'_mask.png'
    # print(mask_path)
    # print(3/0)
    # image_path = data_main_path +"/" + data_name + "_img.png"
    # mask_path = data_main_path +"/" + data_name + "_mask.png"
    # dp_path = data_main_path +"/" + data_name + "_dp.png"
    
    color = np.array(scipy.misc.imread(image_path),dtype='f')
    mask = np.array(scipy.misc.imread(mask_path),dtype='f')
    # dp = np.array(scipy.misc.imread(dp_path),dtype='f')
    
    X = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    X[0,...] = color
    
    Z = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
    Z[0,...,0] = mask>0
    
    # DP = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    # DP[0,...] = dp
    
    Z2C3 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
    Z2C3[...,0]=Z[...,0]
    Z2C3[...,1]=Z[...,0]
    Z2C3[...,2]=Z[...,0]
    
    X = np.where(Z2C3,X,np.ones_like(X)*255.0)
    
    Z3 = Z2C3
    
    # camera
    C = np.zeros((3,4),dtype='f')
    C[0,0]=1
    C[1,1]=1
    C[2,2]=1
    
    R = np.zeros((3,3),dtype='f')
    R[0,0]=1
    R[1,1]=1
    R[2,2]=1
    
    Rt = R
    
    K = np.zeros((3,3),dtype='f')
    K[0,0]=1111.6
    K[1,1]=1111.6

    K[0,2]=960
    K[1,2]=540
    K[2,2]=1
    
    Ki = np.linalg.inv(K)
    cen = np.zeros((3),dtype='f')
    bbs = np.array([[25,477],[420,872],[1,453],[1,453]],dtype='f')
    bbs = np.reshape(bbs,[1,4,2])
    (origin, scaling) = get_origin_scaling(bbs, IMAGE_HEIGHT)
    
    return X,Z, Z3, C, cen,K,Ki, R,Rt,  scaling, origin,point2,data_name

def read_csv(csv_file_arr,f):
    row=1+277*f
    print(row)
    data_name=csv_file_arr[row][0]
    # print(data_name)
    # point=[]
    # print('wwww',data_name,csv_file_arr[row][0].split("/")[7])
    # if data_name == csv_file_arr[row][0].split("/")[7]:
        # print("same!")
    point=csv_file_arr[row+1:row+277]
    return data_name,point

def get_24_joins(point):
    joins=np.zeros(24,2)
    for i in range(23):
        if i==0:
            joins[i,0]=point[i][0]
            joins[i,1]=point[i][1]
            joins[i+1,0]=point[i][2]
            joins[i+1,1]=point[i][3]
        else:
            joins[i+1,0]=point[i][3]
            joins[i+1,0]=point[i][4]
    return joins

def joins_to_png(path,img,joins,num):
    for i in range(joins):
        cv2.circle(img,(joins[i,0],joins[i,1]),10,(0,0,255),3)
        cv2.putText(img,num[i],(joins[i,0]-5,joins[i,1]-5),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
    cv2.imwrite(path,img)

def col_wkdr(point,prediction1):
    wkdr=0
    prediction = (np.array(prediction1[0]))[0,...,0]

    for i in range(276):
        A_x=int(point[i][0])
        A_y=int(point[i][1])
        B_x=int(point[i][2])
        B_y=int(point[i][3])
        ord_gt=int(point[i][4])
        pre_A = prediction[A_x-1,A_y-1]
        pre_B = prediction[B_x-1,B_y-1]  
        if pre_A == pre_B:
            ord_pre=0
        else:
            if pre_A < pre_B:
                ord_pre=1
            else:
                ord_pre=-1
        if ord_gt==ord_pre:
            wkdr+=1
      
    print('order ',A_x,A_y,B_x,B_y,ord_gt,pre_A,pre_B,wkdr)
    return wkdr
# **********************************************************************************************************
def read_test_data_9_channals_tang(data_main_path,data_name,IMAGE_HEIGHT,IMAGE_WIDTH):
    # image_path = data_main_path +"/color_val/" + data_name + ".png"
    # mask_path = data_main_path +"/mask_val/" + data_name + "_mask.png"
    # dp_path = data_main_path +"/densepose_val/" + data_name + "_IUV.png"
    image_path = data_main_path +"/color/" + data_name + ".png"
    mask_path = data_main_path +"/mask/" + data_name + "_mask.png"
    dp_path = data_main_path +"/densepose/" + data_name + "_rgb_IUV.png"

    color = np.array(scipy.misc.imread(image_path),dtype='f')
    mask = np.array(scipy.misc.imread(mask_path),dtype='f')
    dp = np.array(scipy.misc.imread(dp_path),dtype='f')
    
    X = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    X[0,...] = color
    
    Z = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
    Z[0,...,0] = mask>100
    
    DP = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    DP[0,...] = dp
    
    Z2C3 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
    Z2C3[...,0]=Z[...,0]
    Z2C3[...,1]=Z[...,0]
    Z2C3[...,2]=Z[...,0]
    
    X = np.where(Z2C3,X,np.ones_like(X)*255.0)
    
    Z3 = Z2C3
    
    # camera
    C = np.zeros((3,4),dtype='f')
    C[0,0]=1
    C[1,1]=1
    C[2,2]=1
    
    R = np.zeros((3,3),dtype='f')
    R[0,0]=1
    R[1,1]=1
    R[2,2]=1
    
    Rt = R
    
    K = np.zeros((3,3),dtype='f')
    K[0,0]=1111.6
    K[1,1]=1111.6

    K[0,2]=960
    K[1,2]=540
    K[2,2]=1
    
    Ki = np.linalg.inv(K)
    cen = np.zeros((3),dtype='f')
    bbs = np.array([[25,477],[420,872],[1,453],[1,453]],dtype='f')
    bbs = np.reshape(bbs,[1,4,2])
    (origin, scaling) = get_origin_scaling(bbs, IMAGE_HEIGHT)
    
    return X,Z, Z3, DP

def read_test_data_9_channals_thuman(data_main_path,data_name,IMAGE_HEIGHT,IMAGE_WIDTH):
    # image_path = data_main_path +"/color_val/" + data_name + ".png"
    # mask_path = data_main_path +"/mask_val/" + data_name + "_mask.png"
    # dp_path = data_main_path +"/densepose_val/" + data_name + "_IUV.png"
    image_path = data_main_path +"/color/" + data_name + ".png"
    mask_path = data_main_path +"/mask/" + data_name + "_mask.png"
    dp_path = data_main_path +"/densepose/" + data_name + "_IUV.png"
    # normal_name=data_main_path +"/normal_png/" + data_name + "_normal.png"
    color = np.array(scipy.misc.imread(image_path),dtype='f')
    mask = np.array(scipy.misc.imread(mask_path),dtype='f')
    dp = np.array(scipy.misc.imread(dp_path),dtype='f')

    # batch_cur_normal=np.array(scipy.misc.imread(normal_name),dtype='f')
    # cur_normal= np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    # cur_normal[0,:,:,:] = batch_cur_normal
    # n1 = np.genfromtxt(data_main_path  + "/normal_txt/" + data_name+'_NORMAL_1.txt',delimiter=" ")
    # n2 = np.genfromtxt(data_main_path  + "/normal_txt/" + data_name+'_NORMAL_2.txt',delimiter=" ")
    # n3 = np.genfromtxt(data_main_path  + "/normal_txt/" + data_name+'_NORMAL_3.txt',delimiter=" ")
    # cur_normal[...,0] = n1;
    # cur_normal[...,1] = n2;
    # cur_normal[...,2] = n3;
    X = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    X[0,...] = color
    
    Z = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
    Z[0,...,0] = mask>100
    
    DP = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    DP[0,...] = dp
    
    Z2C3 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
    Z2C3[...,0]=Z[...,0]
    Z2C3[...,1]=Z[...,0]
    Z2C3[...,2]=Z[...,0]
    
    X = np.where(Z2C3,X,np.ones_like(X)*255.0)
    
    Z3 = Z2C3
    
    # camera
    C = np.zeros((3,4),dtype='f')
    C[0,0]=1
    C[1,1]=1
    C[2,2]=1
    
    R = np.zeros((3,3),dtype='f')
    R[0,0]=1
    R[1,1]=1
    R[2,2]=1
    
    Rt = R
    
    K = np.zeros((3,3),dtype='f')
    K[0,0]=1111.6
    K[1,1]=1111.6

    K[0,2]=960
    K[1,2]=540
    K[2,2]=1
    
    Ki = np.linalg.inv(K)
    cen = np.zeros((3),dtype='f')
    bbs = np.array([[25,477],[420,872],[1,453],[1,453]],dtype='f')
    bbs = np.reshape(bbs,[1,4,2])
    (origin, scaling) = get_origin_scaling(bbs, IMAGE_HEIGHT)
    
    return X,Z, Z3, DP
def read_test_data_6_channals_tang(data_main_path,data_name,IMAGE_HEIGHT,IMAGE_WIDTH):
    # image_path = data_main_path +"/color/" + data_name + "_rgb.png"
    # mask_path = data_main_path +"/mask/" + data_name + "_mask.png"
    image_path = data_main_path +"/color/" + data_name + "_rgb.png"
    mask_path = data_main_path +"/mask/" + data_name + "_mask.png"
    if os.path.exists(image_path):
        image_path = data_main_path +"/color/" + data_name + "_rgb.png"
        mask_path = data_main_path +"/mask/" + data_name + "_mask.png"
    else:
        image_path = '/data/liqianlin/Evalation_Hdnet/new_test_tang_1310/补充图片/color/' + data_name + ".png"
        mask_path = '/data/liqianlin/Evalation_Hdnet/new_test_tang_1310/补充图片/mask/' + data_name + ".png"

    color = np.array(scipy.misc.imread(image_path),dtype='f')
    mask = np.array(scipy.misc.imread(mask_path),dtype='f')
    
    X = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    X[0,...] = color
    
    Z = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
    Z[0,...,0] = mask>100
    Zb = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH),dtype='b')
    Zb = mask > 100
    Z2C3 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
    Z2C3[...,0]=Z[...,0]
    Z2C3[...,1]=Z[...,0]
    Z2C3[...,2]=Z[...,0]
    
    X = np.where(Z2C3,X,np.ones_like(X)*255.0)
    
    Z3 = Z2C3
    
    # camera
    C = np.zeros((3,4),dtype='f')
    C[0,0]=1
    C[1,1]=1
    C[2,2]=1
    
    R = np.zeros((3,3),dtype='f')
    R[0,0]=1
    R[1,1]=1
    R[2,2]=1
    
    Rt = R
    
    K = np.zeros((3,3),dtype='f')
    K[0,0]=1111.6
    K[1,1]=1111.6

    K[0,2]=960
    K[1,2]=540
    K[2,2]=1
    
    Ki = np.linalg.inv(K)
    cen = np.zeros((3),dtype='f')
    bbs = np.array([[25,477],[420,872],[1,453],[1,453]],dtype='f')
    bbs = np.reshape(bbs,[1,4,2])
    (origin, scaling) = get_origin_scaling(bbs, IMAGE_HEIGHT)
    
    return X,Z, Z3, Zb

def read_test_data_6_channals_thuman(data_main_path,data_name,IMAGE_HEIGHT,IMAGE_WIDTH):
    image_path = data_main_path +"/color/" + data_name + ".png"
    mask_path = data_main_path +"/mask/" + data_name + "_mask.png"
    # image_path = data_main_path +"/color/" + data_name + "_rgb.png"
    # mask_path = data_main_path +"/mask/" + data_name + "_mask.png"
    color = np.array(scipy.misc.imread(image_path),dtype='f')
    mask = np.array(scipy.misc.imread(mask_path),dtype='f')
    
    X = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    X[0,...] = color
    
    Z = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
    Z[0,...,0] = mask>100
    Zb = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH),dtype='b')
    Zb = mask > 100
    Z2C3 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
    Z2C3[...,0]=Z[...,0]
    Z2C3[...,1]=Z[...,0]
    Z2C3[...,2]=Z[...,0]
    
    X = np.where(Z2C3,X,np.ones_like(X)*255.0)
    
    Z3 = Z2C3
    
    # camera
    C = np.zeros((3,4),dtype='f')
    C[0,0]=1
    C[1,1]=1
    C[2,2]=1
    
    R = np.zeros((3,3),dtype='f')
    R[0,0]=1
    R[1,1]=1
    R[2,2]=1
    
    Rt = R
    
    K = np.zeros((3,3),dtype='f')
    K[0,0]=1111.6
    K[1,1]=1111.6

    K[0,2]=960
    K[1,2]=540
    K[2,2]=1
    
    Ki = np.linalg.inv(K)
    cen = np.zeros((3),dtype='f')
    bbs = np.array([[25,477],[420,872],[1,453],[1,453]],dtype='f')
    bbs = np.reshape(bbs,[1,4,2])
    (origin, scaling) = get_origin_scaling(bbs, IMAGE_HEIGHT)
    
    return X,Z, Z3, Zb

def read_test_data_6_channals_thuman_200(data_main_path,filename,data_name,IMAGE_HEIGHT,IMAGE_WIDTH):
    image_path = data_main_path +"/color/" + filename + data_name + ".jpg"
    mask_path = data_main_path +"/mask/" + filename + data_name + ".png"
    color = np.array(scipy.misc.imread(image_path),dtype='f')
    mask = np.array(scipy.misc.imread(mask_path),dtype='f')
    X = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    X[0,...] = color
    Z = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
    Z[0,...,0] = mask>100
    Zb = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH),dtype='b')
    Zb = mask > 100
    Z2C3 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')
    Z2C3[...,0]=Z[...,0]
    Z2C3[...,1]=Z[...,0]
    Z2C3[...,2]=Z[...,0]
    X = np.where(Z2C3,X,np.ones_like(X)*255.0)
    Z3 = Z2C3
    return X,Z, Z3, Zb
# **********************************************************************************************************
def nmap_normalization(nmap_batch):
    image_mag = np.expand_dims(np.sqrt(np.square(nmap_batch).sum(axis=3)),-1)
    image_unit = np.divide(nmap_batch,image_mag)
    return image_unit
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst    
# **********************************************************************************************************   
def path_leaf(inpath):
    head, tail = ntpath.split(inpath)
    return tail or ntpath.basename(head)
def get_test_data_9_channals_tang(inpath):
    pngpath = inpath+'/color_wobg/*.png'
    all_img  = glob.glob(pngpath)
    print('len(all_img)',len(all_img))
    filename_list = []
    for i in range(len(all_img)):
        img_name = path_leaf(all_img[i])
        # print(img_name)
        # print(3/0)
        name = img_name.split('_rgb')[0]
        dpname = name+"_rgb_IUV.png"
        mname = name+"_mask.png"
        # print(dpname,mname)
        if path.exists(inpath+'/densepose/'+dpname) and path.exists(inpath+'/mask/'+mname):
            filename_list.append(name)       
    return filename_list

def get_test_data_9_channals_thuman(inpath):
    pngpath = inpath+'/color/*.png'
    all_img  = glob.glob(pngpath)
    print('len(all_img)',len(all_img))
    filename_list = []
    for i in range(len(all_img)):
        img_name = path_leaf(all_img[i])
        # print(img_name)
        # print(3/0)
        name = img_name.split('.')[0]
        dpname = name+"_IUV.png"
        mname = name+"_mask.png"
        # print(dpname,mname)
        if path.exists(inpath+'/densepose/'+dpname) and path.exists(inpath+'/mask/'+mname):
            filename_list.append(name) 
            
    return filename_list
def get_test_data_6_channals_tang(apath):
    pngpath = apath+'/color/*.png'
    all_img  = glob.glob(pngpath)
    print('len(all_img)',len(all_img))
    filename_list = []
    for i in range(len(all_img)):
        img_name = path_leaf(all_img[i])
        # print('img_name',img_name)
        # print(3/0)
        # name = img_name.split('_rgb')[0]
        name = img_name.split('.')[0]
        mname = name+"_mask.png"
        # print('mname,name: ',mname,name)  
        if path.exists(apath+'/mask/'+mname):
            filename_list.append(name) 
              
    return filename_list

def get_test_data_6_channals_thuman(apath):
    pngpath = apath+'/color/*.png'
    all_img  = glob.glob(pngpath)
    print('len(all_img)',len(all_img))
    filename_list = []
    for i in range(len(all_img)):
        img_name = path_leaf(all_img[i])
        # print('img_name',img_name)
        # print(3/0)
        # name = img_name.split('_rgb')[0]
        name = img_name.split('.')[0]
        mname = name+"_mask.png"
        # print('mname,name: ',mname,name)  
        if path.exists(apath+'/mask/'+mname):
            filename_list.append(name) 
              
    return filename_list
# **********************************************************************************************************  
# Function borrowed from https://github.com/sfu-gruvi-3dv/deep_human
def depth2mesh(depth, mask, filename):
    h = depth.shape[0]
    w = depth.shape[1]
    depth = depth.reshape(h,w,1)
    f = open(filename + ".obj", "w")
    for i in range(h):
        for j in range(w):
            f.write('v '+str(float(2.0*i/h))+' '+str(float(2.0*j/w))+' '+str(float(depth[i,j,0]))+'\n')

    threshold = 0.07

    for i in range(h-1):
        for j in range(w-1):
            if i < 2 or j < 2:
                continue
            localpatch= np.copy(depth[i-1:i+2,j-1:j+2])
            dy_u = localpatch[0,:] - localpatch[1,:]
            dx_l = localpatch[:,0] - localpatch[:,1]
            dy_d = localpatch[0,:] - localpatch[-1,:]
            dx_r = localpatch[:,0] - localpatch[:,-1]
            dy_u = np.abs(dy_u)
            dx_l = np.abs(dx_l)
            dy_d = np.abs(dy_d)
            dx_r = np.abs(dx_r)
            if np.max(dy_u)<threshold and np.max(dx_l) < threshold and np.max(dy_d) < threshold and np.max(dx_r) < threshold and mask[i,j]:
                f.write('f '+str(int(j+i*w+1))+' '+str(int(j+i*w+1+1))+' '+str(int((i + 1)*w+j+1))+'\n')
                f.write('f '+str(int((i+1)*w+j+1+1))+' '+str(int((i+1)*w+j+1))+' '+str(int(i * w + j + 1 + 1)) + '\n')
    f.close()
    return
