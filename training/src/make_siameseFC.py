import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')  #'../'代表上一级目录
from src.convolutional import set_convolutional
from src.create_label import create_label
from src.crops import pad_frame,extract_crops

#the follow parameters have to reflect the design of the network to be imported
#原文章中用了两个GPU，此处尝试不用
_conv_w_sz=np.array([11,5,3,3,3]) #the map size of filter(weight of conv net)
_conv_w_in_c=np.array([3,96,256,384,384])# the input channle number of filter
_conv_w_out=np.array([96,256,384,384,256])# the output number of feature map(the number of filter kernel)
_conv_stride=np.array([2,1,1,1,1])
_pool_stride=np.array([2,2,0,0,0])#0 means no pooling
_pool_sz=np.array([3,3,0,0,0])#0 means no pooling
_if_bnorm=np.array([0,0,0,0,0],dtype=bool)# if batchnorm
_if_relu=np.array([1,1,1,1,0],dtype=bool)
_nums_layers=len(_conv_w_sz)#the number of layers

pos_x=tf.placeholder(tf.float64)
pos_y=tf.placeholder(tf.float64)
#从原图中裁剪出的z和x的size（非最后训练尺寸）
#the size of z and x from origin image,not the last training size
z_size=tf.placeholder(tf.float64)
x_size=tf.placeholder(tf.float64)


def make_siameseFC(env,design,hp):
    #-------------------------------------------------------------------------
    #function//im_z和im_x可复用此函数
    #-------------------------------------------------------------------------
    filename=tf.placeholder(tf.string,[],name='filename')
    image_file=tf.read_file(filename)
    
    #Decode the image as a JPEG/BMP... file,and turn it into a tensor
    #choose the decode type
    if env.image_type == 'jpg':
        image=tf.image.decode_jpeg(image_file)
    elif env.image_type == 'bmp':
        image=tf.image.decode_bmp(image_file)
    
    #将像素值缩放到[0,1]
    im=255.0*tf.image.convert_image_dtype(image,tf.float32)
    
    frame_size=tf.shape(im)
    
    if design.pad_with_image_mean:
        #get the mean pixel value of each channle
        avg_chan=tf.reduce_mean(im,axis=(0,1),name='avg_chan')
    else:
        avg_chan=None
        
    #pad the image before crop
    #def pad_frame(im,frame_size,pos_x,pos_y,patch_size,avg_chan)
    #z
    im_padded_z,npad_z=pad_frame(im,frame_size,pos_x,pos_y,z_size,avg_chan)
    im_padded_z=tf.cast(im_padded_z,tf.float32)
    #crop the z patch
    #def extract_crops(im,npad,pos_x,pos_y,size_src,size_dst)
    crop_z=extract_crops(im_padded_z,npad_z,pos_x,pos_y,z_size,design.exemplarSize)
    #x
    print(x_size)
    im_padded_x,npad_x=pad_frame(im,frame_size,pos_x,pos_y,x_size,avg_chan)
    im_padded_x=tf.cast(im_padded_x,tf.float32)
    #crop the x patch
    crop_x=extract_crops(im_padded_x,npad_x,pos_x,pos_y,x_size,design.instacneSize)
    
    #use the crops as a input of Siamese net to train
    _siam_net_z,_siam_net_x=create_net(crop_x,crop_z)
    #evaliate the correlation between x and z
    scores=_match_templates(_siam_net_z,_siam_net_x)
    #upsample the score maps
    scores_up=tf.image.resize_images(scores,[design.score_size,design.score_size],
                                     method=tf.image.ResizeMethod.BICUBIC,align_corners=True)
    
    scores_gt=create_label([design.score_size,design.score_size],design.dPos)
    
    Hz,Wz,Bz,Cz=tf.unstack(tf.shape(scores_up))
    scores_up_re=tf.squeeze(tf.reshape(scores_up,(1,1,1,Hz*Wz*Bz*Cz)))
    scores_gt_re=tf.squeeze(tf.reshape(scores_gt,(1,1,1,Hz*Wz*Bz*Cz)))
    
    #train --back propagation
    #if need tf.sqrt???????????????
    print('begin calculate the loss')
    loss=tf.sqrt(tf.reduce_mean(tf.square(scores_up_re-tf.cast(scores_gt_re,tf.float32))))
    #loss=_siam_net_z-1
    #train --back propagation
    #the train_op trains the variables that define with "tf.Variable" or "tf.get_variable"
    train_op=tf.train.AdamOptimizer(hp.learning_rate).minimize(loss)
    print('loss end1')
    
    return filename,_siam_net_z,loss,train_op
    
def init_create_net():
    for i in range(_nums_layers,):
        scope_name='conv'+str(i+1)
        with tf.variable_scope(scope_name or 'conv'):
            W=tf.get_variable("W",[_conv_w_sz[i],_conv_w_sz[i],_conv_w_in_c[i],_conv_w_out[i]],
                               trainable=False,initializer=tf.truncated_normal_initializer(stddev=0.1))
            b=tf.get_variable("b",_conv_w_out[i],
                              trainable=False,initializer=tf.truncated_normal_initializer(stddev=0.1))
            
        
def create_net(net_x,net_z):
    #-------------------------------------------------------------------------
    #function//net_x:instance frame ;net_z:template frame
    #-------------------------------------------------------------------------
    #not sure
    #W_param_list=[n for n in range(0,_nums_layers)]
    #b_param_list=[n for n in range(0,_nums_layers)]
    for i in range(_nums_layers):
        print('Layer '+str(i+1))
        
        #set up the conv bolck
        #set_convolutional(X,stride,bn_beta,bn_gamma,bn_init_mean,bn_init_var,batchnorm=True,activation=True,reuse=False,scope=None):
        print(net_z)
        print(net_x)
        net_x=set_convolutional(net_x,
                                [_conv_w_sz[i],_conv_w_sz[i],_conv_w_in_c[i],_conv_w_out[i]],_conv_w_out[i],#the shape of W and b
                                _conv_stride[i],0,0,0,0,batchnorm=False,
                                activation=_if_relu[i],reuse=True,scope='conv'+str(i+1))
        
        net_z=set_convolutional(net_z,
                                [_conv_w_sz[i],_conv_w_sz[i],_conv_w_in_c[i],_conv_w_out[i]],_conv_w_out[i],#the shape of W and b
                                _conv_stride[i],0,0,0,0,batchnorm=False,
                                activation=_if_relu[i],reuse=True,scope='conv'+str(i+1))
        print(net_z)
        print(net_x)
        print('Layer '+str(i+1)+' conv end')
        #if having the pooling
        if _pool_stride[i]>0:
            print("_pool_stride")
            net_x=tf.nn.max_pool(net_x,[1,_pool_sz[i],_pool_sz[i],1],strides=[1,_pool_stride[i],_pool_stride[i],1],
                                 padding='VALID',name='pool'+str(i+1))
            net_z=tf.nn.max_pool(net_z,[1,_pool_sz[i],_pool_sz[i],1],strides=[1,_pool_stride[i],_pool_stride[i],1],
                                 padding='VALID',name='pool'+str(i+1))
        
        print(net_z)
        print(net_x)
        print('Layer '+str(i+1)+' end')
    
    return net_z,net_x

def create_net_define_var(crop_x,crop_z):
    with tf.variable_scope('conv1'):
    #with tf.variable_scope(scope or 'conv'):
        #trainable:标记是否加入GraphKeys.TRAINABLE_VARIABLES集合
        #tf.truncated_normal_initializer(stddev=0.1):生成的随机的标准方差*********以高斯分布的方式初始化W和b，之后复用（reuse=True)
        W=tf.get_variable("W",[_conv_w_sz[0],_conv_w_sz[0],_conv_w_in_c[0],_conv_w_out[0]],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b=tf.get_variable("b",_conv_w_out[0],initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        #padding='VALID'：按照(图片大小-filterSize(=W.size))/stride+1
        #padding='SAME' :大小和原图像一致
        #stride:卷积的步长
        stride=2
        h_x=tf.nn.conv2d(crop_x,W,strides=[1,stride,stride,1],padding='VALID')+b
        
      
        h_x=tf.nn.relu(h_x)
            
        h_z=tf.nn.conv2d(crop_z,W,strides=[1,stride,stride,1],padding='VALID')+b
        
        h_z=tf.nn.relu(h_z)
            
        return h_z,h_x

def _match_templates(net_z,net_x):
    #-------------------------------------------------------------------------
    #function//use the result (x and z) from conv layers to evaluate the correlation between x and z
    #-------------------------------------------------------------------------
    print('match_template')
    #z,x,are [Batch(num),H,W,C]
    net_z=tf.transpose(net_z,perm=[1,2,0,3])
    net_x=tf.transpose(net_x,perm=[1,2,0,3])
    #after transpose,z,x are[H,W,B,C]
    
    #get the num of each dimension of net_x and net_z
    Hz,Wz,Bz,Cz=tf.unstack(tf.shape(net_z))
    Hx,Wx,Bx,Cx=tf.unstack(tf.shape(net_x))
    #assert Bz==Bx ('Z and X must have the same Batch Size ')
    #assert Cz==Cx ('Z and X must have the same Channel numbers')
    
    #filter(W) shape type:[f_h,f_w,in_channels,channel_mulitpler(the num of filter)]
    net_z=tf.reshape(net_z,(Hz,Wz,Bz*Cz,1))
    #the conv input data shape type:[batch_num,in_h,in_w,in_channels]
    net_x=tf.reshape(net_x,(1,Hx,Wx,Bx*Cx))
    
    net_final=tf.nn.depthwise_conv2d(net_x,net_z,strides=[1,1,1,1],padding='VALID')
    #final is [1,Hf,Wf,BC]
    #net_final=tf.concat(tf.split(net_final,3,axis=3),axis=0)
    #final is [B,Hf,Wf,C]
    #tf.reduce_sum()求和后会降维，需要用tf.expand_dims在axis=3处增加一维
    net_final=tf.expand_dims(tf.reduce_sum(net_final,axis=3),axis=3)
    #final is [B,Hf,Wf,1]
    
    return net_final