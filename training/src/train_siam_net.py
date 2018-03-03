import tensorflow as tf
import os
import numpy as np
import time
import src.make_siameseFC as siam
from src.region_to_bbox import region_to_bbox
from PIL import Image

def train_siam_net(design,hp,frame_name_list,num_frames,gt,filename,conv_W,conv_b,siam_net_z,loss,train_op):
    #-------------------------------------------------------------------------
    #index_z:the index of template in the frame_name_list
    #-------------------------------------------------------------------------
    
    with tf.Session() as sess:
        #tf.global_variables_initializer().run()
        sess.run(tf.global_variables_initializer())
        #tf.local_variables_initializer().run()
        #Coordinate the loading of image files
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        
        #TB
        merged=tf.summary.merge_all()
        writer=tf.summary.FileWriter('/tmp/tensorlogs/siamtf',sess.graph)
        
        for i in range(0,num_frames-1):
            pos_x,pos_y,target_w,target_h=region_to_bbox(gt[i])
             #connect the context to get the size of x and z crops
            t_sz=(target_w+target_h)*design.context_amount
            w_crop_z=target_w+t_sz
            h_crop_z=target_h+t_sz
            sz_z=np.sqrt(float(w_crop_z)*float(h_crop_z))
            sz_x=float(design.instacneSize)/float(design.exemplarSize)*sz_z
            
            siam_net_z_ = sess.run([siam_net_z],feed_dict={
            #sess.run([train_op],feed_dict={
                                                       siam.pos_x:pos_x,
                                                       siam.pos_y:pos_y,
                                                       siam.z_size:sz_z,
                                                       filename:frame_name_list[i]})
        
            #t_start=time.time()
            #print('begin')
            #train the image which is the pair of siam_net_z
            result,train_op_=sess.run([merged,train_op],feed_dict={
                                              siam.pos_x:pos_x,
                                              siam.pos_y:pos_y,
                                              siam.z_size:sz_z,
                                              siam.x_size:float(sz_x),
                                              siam_net_z:siam_net_z_[0],
                                              filename:frame_name_list[i+1] })
            
            writer.add_summary(result,i)
        #print('loss end')
        #train --back propagation
        #tf.train.AdamOptimizer(hp.learning_rate).minimize(loss_)
           
        
        coord.request_stop()
        coord.join(threads)
        
      
        
        
    