import tensorflow as tf
import numpy as np

#填充图像（防止追踪时裁剪的矩形框超出原来的图像）
def pad_frame(im,frame_size,pos_x,pos_y,patch_size,avg_chan):
    c=patch_size/2
    
    xleft_pad=tf.maximum(0,-tf.cast(tf.round(pos_x-c),tf.int32))
    ytop_pad=tf.maximum(0,-tf.cast(tf.round(pos_y-c),tf.int32))
    #frame_size[0]->y,frame_size[1]->x
    xright_pad=tf.maximum(0,tf.cast(tf.round(pos_x+c),tf.int32)-frame_size[1])
    ybottom_pad=tf.maximum(0,tf.cast(tf.round(pos_y+c),tf.int32)-frame_size[0])
    
    #取四个方向中最大的值进行pad
    npad=tf.reduce_max([xleft_pad,ytop_pad,xright_pad,ybottom_pad])
    paddings=[[npad,npad],[npad,npad],[0,0]]
    im_padded=im
    
    if avg_chan is not None:
        im_padded=im_padded-avg_chan
    
    im_padded=tf.pad(im_padded,paddings,mode='CONSTANT')
    
    if avg_chan is not None:
        im_padded=im_padded+avg_chan
    
    return im_padded,npad

def extract_crops(im,npad,pos_x,pos_y,size_src,size_dst):
    #提取patch的中心
    c=size_src/2
    #经过pad后的crop的左上角坐标
    rect_lx=npad+tf.cast(tf.round(pos_x-c),tf.int32)
    rect_ly=npad+tf.cast(tf.round(pos_y-c),tf.int32)
    width=tf.round(pos_x+c)-tf.round(pos_x-c)
    height=tf.round(pos_y+c)-tf.round(pos_y-c)
    #对图像进行裁剪
    crop=tf.image.crop_to_bounding_box(im,
                                      tf.cast(rect_ly,tf.int32),
                                      tf.cast(rect_lx,tf.int32),
                                      tf.cast(height,tf.int32),
                                      tf.cast(width,tf.int32))
    crop=tf.image.resize_images(crop,[size_dst,size_dst],method=tf.image.ResizeMethod.BILINEAR)
    #在第0维增加一个维度，值为1，使其单独成为一张图片
    crops=tf.expand_dims(crop,axis=0)
    
    return crops

