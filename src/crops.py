from __future__ import division
import numpy as np
import tensorflow as tf
from PIL import Image
import functools

def resize_images(images,size,resample):
    #'''Alternative to tf.image.resize_images that uses PIL'''
    fn=functools.partial(_resize_image,size=size,resample=resample)
    #py_func(func,input,Tout,stateful=True,name=None)
    return tf.py_func(fn,[images],images.dtype)

def _resize_images(x,size,resample):
    if len(x.shape)==3:
        return _resize_image(x,size,resample)
    assert len(x.shape)==4
    y=[]
    for i in range(x.shape[0]):
        y.append(_resize_image(x[i]))
    y=np.stack(y,axis=0)
    return y
        
def _resize_image(x,size,resample):
    #当条件为false则引发异常
    assert len(x.shape)==3
    
    y=[]
    for j in range(x.shape[2]):
        #f.shape=(width,height,1)
        f=x[:,:,j]
        print (f.shape)
        #将图像单个通道的array转化成image
        f=Image.fromarray(f)
        #注意resize尺寸的顺序
        f=f.resize((size[1],size[0]),resample=resample)
        f=np.array(f)
        y.append(f)
    #将RGB的三个单通道array重新堆叠成三通道(shape:width,height,c)
    return np.stack(y,axis=2)

#填充图像(防止追踪时使用的矩形框超出原来的图像范围，当超出范围是进行适当的填充)
def pad_frame(im,frame_sz,pos_x,pos_y,patch_sz,avg_chan):
    c=patch_sz/2
    
    #tf.cast(value,dtype):强制类型转换
    #tf.round(value):取整
    #pos_x-c:若为负值说明超出了原来的图像的范围，再去负数，得到正数，即为需要填充的部分，否则为0，不需填充。
    xleft_pad=tf.maximum(0,-tf.cast(tf.round(pos_x-c),tf.int32))
    ytop_pad=tf.maximum(0,-tf.cast(tf.round(pos_y-c),tf.int32))
    #image的shape问题
    xright_pad=tf.maximum(0,tf.cast(tf.round(pos_x+c),tf.int32)-frame_sz[1])
    ybottom_pad=tf.maximum(0,tf.cast(tf.round(pos_y+c),tf.int32)-frame_sz[0])
    #取四个方向中pad部分的最大值
    npad=tf.reduce_max([xleft_pad,ytop_pad,xright_pad,ybottom_pad])
    paddings=[[npad,npad],[npad,npad],[0,0]]
    im_padded=im
    #why this?
    if avg_chan is not None:
        im_padded=im_padded - avg_chan
    im_padded=tf.pad(im_padded,paddings,mode='CONSTANT')
    if avg_chan is not None:
        im_padded=im_padded + avg_chan
    return im_padded,npad

    #图像裁剪
def extract_crops_z(im,npad,pos_x,pos_y,sz_src,sz_dst):
    c=sz_src/2
    #get the postition of the box and consider padding
    tr_x=npad+tf.cast(tf.round(pos_x-c),tf.int32)
    tr_y=npad+tf.cast(tf.round(pos_y-c),tf.int32)
    width=tf.round(pos_x+c)-tf.round(pos_x-c)
    height=tf.round(pos_y+c)-tf.round(pos_y-c)
    #对图像进行裁剪
    crop=tf.image.crop_to_bounding_box(im,
                                      tf.cast(tr_y,tf.int32),
                                      tf.cast(tr_x,tf.int32),
                                      tf.cast(height,tf.int32),
                                      tf.cast(width,tf.int32))
    crop=tf.image.resize_images(crop,[sz_dst,sz_dst],method=tf.image.ResizeMethod.BILINEAR)
    #在第0维增加一个维度，值为1，使其单独成为一张图片
    crops=tf.expand_dims(crop,axis=0)
    return crops

def extract_crops_x(im,npad,pos_x,pos_y,sz_src0,sz_src1,sz_src2,sz_dst):
    #take center of the biggest scaled source path
    c=sz_src2/2
    #get top right corner of bbox and consider padding
    tr_x=npad+tf.cast(tf.round(pos_x-c),tf.int32)
    tr_y=npad+tf.cast(tf.round(pos_y-c),tf.int32)
    width=tf.round(pos_x+c)-tf.round(pos_x-c)
    height=tf.round(pos_y+c)-tf.round(pos_y-c)
    search_area=tf.image.crop_to_bounding_box(im,
                                             tf.cast(tr_y,tf.int32),
                                             tf.cast(tr_x,tf.int32),
                                             tf.cast(height,tf.int32),
                                             tf.cast(width,tf.int32))
    
    offset_s0=(sz_src2-sz_src0)/2
    offset_s1=(sz_src2-sz_src1)/2
   
    #利用crop得到的最大patch s2继续裁减较小的s0和s1
    crop_s0=tf.image.crop_to_bounding_box(search_area,
                                         tf.cast(offset_s0,tf.int32),
                                         tf.cast(offset_s0,tf.int32),
                                         tf.cast(tf.round(sz_src0),tf.int32),
                                         tf.cast(tf.round(sz_src0),tf.int32))
    crop_s0=tf.image.resize_images(crop_s0,[sz_dst,sz_dst],method=tf.image.ResizeMethod.BILINEAR)
    crop_s1=tf.image.crop_to_bounding_box(search_area,
                                         tf.cast(offset_s1,tf.int32),
                                         tf.cast(offset_s1,tf.int32),
                                         tf.cast(tf.round(sz_src1),tf.int32),
                                         tf.cast(tf.round(sz_src1),tf.int32))
    crop_s1=tf.image.resize_images(crop_s1,[sz_dst,sz_dst],method=tf.image.ResizeMethod.BILINEAR)
    crop_s2=tf.image.resize_images(search_area,[sz_dst,sz_dst],method=tf.image.ResizeMethod.BILINEAR)
    crops=tf.stack([crop_s0,crop_s1,crop_s2])
    return crops
    