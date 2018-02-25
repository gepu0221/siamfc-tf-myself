import tensorflow as tf
import os
import numpy as np
import time
import src.make_siameseFC as siam
from PIL import Image

def train_siam_net(design,hp,frame_name_list,z_index,pos_x,pos_y,target_w,target_h,filename,siam_net_z,loss,train_op):
    #-------------------------------------------------------------------------
    #index_z:the index of template in the frame_name_list
    #-------------------------------------------------------------------------
    
    #connect the context to get the size of x and z crops
    t_sz=(target_w+target_h)*design.context_amount
    w_crop_z=target_w+t_sz
    h_crop_z=target_h+t_sz
    sz_z=np.sqrt(w_crop_z*h_crop_z)
    sz_x=design.instacneSize/design.exemplarSize*sz_z
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #Coordinate the loading of image files
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        
        siam_net_z_ = sess.run([siam_net_z],feed_dict={
                                                       siam.pos_x:pos_x,
                                                       siam.pos_y:pos_y,
                                                       siam.z_size:sz_z,
                                                       filename:frame_name_list[z_index]})
        
        #t_start=time.time()
        print('begin')
        #train the image which is the pair of siam_net_z
        sess.run([train_op],feed_dict={
                                              siam.pos_x:pos_x,
                                              siam.pos_y:pos_y,
                                              siam.x_size:sz_x,
                                              siam_net_z:siam_net_z_[0],
                                              filename:frame_name_list[z_index+1] })
        print('loss end')
        #train --back propagation
        #tf.train.AdamOptimizer(hp.learning_rate).minimize(loss_)
        
        
        
    