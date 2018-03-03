import sys
import os 
import numpy as np
import time
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework import graph_util  
from tensorflow.python.platform import gfile  
#myself
import src.make_siameseFC as siam
from src.train_siam_net import train_siam_net
from src.region_to_bbox import region_to_bbox
from src.parse_arguments import parse_arguments

_conv_w_sz=np.array([11,5,3,3,3]) #the map size of filter(weight of conv net)
_conv_w_in_c=np.array([3,96,256,384,384])# the input channle number of filter
_conv_w_out=np.array([96,256,384,384,256])# the output number of feature map(the number of filter kernel)

def main():
    #avoid printing TF debugging information(only show error log)
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    hp,evaluation,run,env,design=parse_arguments()
    #build TF graph in siamese once for all
    #siam.init_create_net()
    filename,siam_net_z,loss,train_op=siam.make_siameseFC(env,design,hp)
    
    #iterate through all videos of evaluation.dataset
    if evaluation.video == 'all':
        #the path of folder of all videos
        train_data_folder=os.path.join(env.root_train_dataset,evaluation.dataset)
        videos_list=[v for v in os.listdir(train_data_folder)]
        videos_list.sort()
        num_v=len(videos_list)
        for i in range(num_v):
            gt,frame_name_list,frame_sz,n_frames=_init_train_video(env,evaluation,videos_list[i])
            start_frame=evaluation.start_frame
            #not sure
            #gt_=gt[start_frame:,:]
            gt_=gt[start_frame:]
            frame_name_list_=frame_name_list[start_frame:]
            num_frames=np.size(frame_name_list_)
            
            for j in range(num_frames-1):
                pos_x,pos_y,target_w,target_h=region_to_bbox(gt_[j])
                #train_siam_net(design,hp,frame_name_list,z_index,pos_x,pos_y,target_w,target_h,filename,siam_net_z,loss)
                train_siam_net(design,hp,frame_name_list,j,pos_x,pos_y,target_w,target_h,filename,siam_net_z,loss,train_op)
        
    else:
        gt,frame_name_list,_,_ = _init_train_video(env,evaluation,evaluation.video)
        start_frame=evaluation.start_frame
        gt_=gt[start_frame:]
        frame_name_list_=frame_name_list[start_frame:]
        num_frames=np.size(frame_name_list_)
       
        train_siam_net(design,hp,frame_name_list,num_frames,gt,filename,siam_net_z,loss,train_op)
        '''for i in range(num_frames-1):
            pos_x,pos_y,target_w,target_h=region_to_bbox(gt[evaluation.start_frame])
            train_siam_net(design,hp,frame_name_list,i,pos_x,pos_y,target_w,target_h,filename,siam_net_z,loss,train_op)'''
            
    #write_file_param(design,env,evaluation)
    
def _init_train_video(env,evaluation,video):
     #-------------------------------------------------------------------------
    #function//init info of a train video sequence
    #-------------------------------------------------------------------------
    #the path of train_data folder
    train_data_folder=os.path.join(env.root_train_dataset,evaluation.dataset,video)
    #os.listdir():show the file list of the folder
    frame_name_list=[f for f in os.listdir(train_data_folder) if f.endswith("."+env.image_type)]
    frame_name_list=[os.path.join(train_data_folder,'')+s for s in frame_name_list]
    frame_name_list.sort()
    
    #get the info of first frame
    with Image.open(frame_name_list[0]) as img:
        frame_sz=np.asarray(img.size)
        ##????????????????????????????????????????????
        frame_sz[1],frame_sz[0]=frame_sz[0],frame_sz[1]
        
        #read the initialization from ground truth(init the template_z)
    gt_file=os.path.join(train_data_folder,evaluation.gt_name)
    gt=np.genfromtxt(gt_file,delimiter=evaluation.gt_delimiter)
    num_frame=len(frame_name_list)
    assert num_frame == len(gt),'number of frame and number of gt should be the same'
    
    return gt,frame_name_list,frame_sz,num_frame
    
def write_file_param(design,env,evaluation):
    _layers_num=design.layers_num
    
    #get the time stamp now
    time_s=time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
    re_filename=str(time_s)+evaluation.train_re_param_file
    re_file_path=os.path.join(env.root_train_result_param,re_filename)
    for i in range(_layers_num):
        scope_name='conv'+str(i+1)
        with tf.variable_scope(scope_name,reuse=True):
            W_=tf.get_variable("W",shape=[_conv_w_sz[i],_conv_w_sz[i],_conv_w_in_c[i],_conv_w_out[i]])
            b_=tf.get_variable("b",shape=[_conv_w_out[i]])
            w_name='W'+str(i)
            b_name='b'+str(i)
            W_=tf.get_variable(w_name,shape=[_conv_w_sz[i],_conv_w_sz[i],_conv_w_in_c[i],_conv_w_out[i]],tf.constant_initalizer(W_))
            b_=tf.get_variable(b_name,shape=[_conv_w_out[i]],tf.constant_initalizer(b_))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(_layers_num):
            w_name='W'+str(i)
            b_name='b'+str(i)
            graph_w=graph_util.convert_variables_to_constants(sess,sess.graph_def,[w_name])
            tf.train.wirte_graph(graph_w,'.',re_file_path,as_text=False)
            graph_b=graph_util.convert_variables_to_constants(sess,sess.graph_def,[b_name])
            tf.train.write_graph(graph_b,'.',re_file_path,as_text=False)
            
if __name__=='__main__':
    sys.exit(main())