from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox

def main():
    #avoid printing TF debugging information
    #仅显示error log
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    #TODO:allow parameters from command line or leave everything in json files?
    hp,evaluation,run,env,design=parse_arguments()
    
    
    #gt_,frame_name_list_,_,_=_init_video(env,evaluation,evaluation.video)
    #pos_x,pos_y,target_w,target_h=region_to_bbox(gt_[0])
    #print('---target_w---'+"%d"%target_w+'--target_h---'+"%d"%target_h)
    #why????????????? 
    #Set size for use with tf.image.resize_images with align_corners=True
    #For example:
    # [1,4,7]=>[1 2 3 4 5 6 7]  (length 3*(3-1)+1)
    #instead of
    #[1,4,7]=>[1 1 2 3 4 5 6 7 7](length 3*3)
    #Why hp.response_up???
    #design.score_sz=33
    #hp.response_up=8
    final_score_sz=hp.response_up*(design.score_sz-1)+1
    #build TF graph once for all
    #filename,image,templates_z,scores are only processes.!!!
    #真正返回信息需要用sess去执行（tracker中执行）
    #return filename, image, templates_z, scores_up
    filename,image,templates_z,scores=siam.build_tracking_graph(final_score_sz,design,env)
    
    #iterate through all videos of evaluation dataset
    if evaluation.video=='all':
        dataset_folder=os.path.join(env.root_dataset,evaluation.dataset)
        #os.listdir(path):返回指定路径下的文件和文件夹
        videos_list=[v for v in os.listdir(dataset_folder)]
        videos_list.sort()
        nv=np.size(videos_list)
        speed=np.zeros(nv*evaluation.n_subseq)
        precisions=np.zeros(nv*evaluation.n_subseq)
        precisions_auc=np.zeros(nv*evaluation.n_subseq)
        ious=np.zeros(nv*evaluation.n_subseq)
        lengths=np.zeros(nv*evaluation.n_subseq)
        #遍历不同的视频样本
        for i in range(nv):
            #frame_name_list:each image of a video sequence
            gt,frame_name_list,frame_sz,n_frames=_init_video(env,evaluation,videos_list[i])
            #np.rint():对浮点数取整但不改变浮点数类型
            #n_subseq=3
            starts=np.rint(np.linspace(0,n_frame-1,evaluation.n_subseq+1))
            #分成n_subseq+1份，将数组赋给starts
            starts=starts[0:evaluation.n_subseq]
            for j in range(evaluation.n_subseq):
                start_frame=int(starts[j])
                #start_frame:指start_frame及以后(选取了n_subseq中的一份)
                gt_=gt[start_frame:,:]
                frame_name_list_=frame_name_list[start_frame:]
                pos_x,pos_y,target_w,target_h=region_to_bbox(gt_[0])
                idx=i*evaluation.n_subseq+j
                #Update
                bboxes,speed[idx]=tracker(hp,run,design,env,evaluation,frame_name_list_,pos_x,pos_y,
                #bboxes,speed[idx]=tracker(hp,run,design,frame_name_list_,pos_x,pos_y,
                                         target_w,target_h,final_score_sz,filename,
                                         image,templates_z,scores,start_frame)
                #gt_:ground truth
                #bboxes:the result of tracking
                lengths[idx],precisions[idx],precisions_auc[idx],ious[idx]=_compile_results(gt_,bboxes,
                                                                                            evaluation.dist_threshold)
                print(str(i)+'--'+videos_list[i]+
                      '--Precision: '+"%.2f"%precisions[idx]+
                      '--Precisions AUC: '+"%.2f"%precisions_auc[idx]+
                      '--IOU: '+"%.2f"%ious[idx]+
                      '--Speed: '+"%.2f"%speed[idx]+'--')
    
    else:
        #evaluation.video='all'
        print(evaluation.video)
        gt,frame_name_list,_,_=_init_video(env,evaluation,evaluation.video)
        #evaluation.start_frame=0
        pos_x,pos_y,target_w,target_h=region_to_bbox(gt[evaluation.start_frame])
        
        #Update
        #bboxes,speed=tracker(hp,run,design,frame_name_list,pos_x,pos_y,target_w,target_h,final_score_sz,
        bboxes,speed=tracker(hp,run,design,env,evaluation,frame_name_list,pos_x,pos_y,target_w,target_h,final_score_sz,
                            filename,image,templates_z,scores,evaluation.start_frame)
        _,precision,precisions_auc,iou=_compile_results(gt,bboxes,evaluation.dist_threshold)
         #print(evaluation.video+
        print(evaluation.video+'--Precision: '+"(%d px)"%evaluation.dist_threshold+': '+"%.2f"%precision+
                      '--Precisions AUC: '+"%.2f"%precisions_auc+
                      '--IOU: '+"%.2f"%iou+
                      '--Speed: '+"%.2f"%speed+'--')
        
#init a video info of a video sequence
def _init_video(env,evaluation,video):
    video_folder=os.path.join(env.root_dataset,evaluation.dataset,video)
    #get the each image from image file
    #Update
    frame_name_list=[f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    #frame_name_list=[f for f in os.listdir(video_folder) if f.endswith(".bmp")]
    frame_name_list=[os.path.join(env.root_dataset,evaluation.dataset,video,'')+s for s in frame_name_list]
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz=np.asarray(img.size)
        #????img的width和height的顺序????????????
        frame_sz[1],frame_sz[0]=frame_sz[0],frame_sz[1]
        
    #read the initialization from ground truth
    gt_file=os.path.join(video_folder,'groundtruth.txt')
    #将txt转换为表格
    #Update
    gt=np.genfromtxt(gt_file,delimiter=',')
    #gt=np.genfromtxt(gt_file,delimiter=' ')
    #the frame num of video
    n_frames=len(frame_name_list)
    print(n_frames)
    print(len(gt))
    assert n_frames==len(gt),'Number of frames and number of GT lines should be equal.'
    
    return gt,frame_name_list,frame_sz,n_frames

def _compile_results(gt,bboxes,dist_threshold):
    l=np.size(bboxes,0)
    #np.zeros(shape=(1,4),dtype=float, order='C')
    gt4=np.zeros((l,4))
    new_distances=np.zeros(l)
    new_ious=np.zeros(l)
    n_thresholds=50
    precisions_ths=np.zeros(n_thresholds)

    for i in range(l):
        gt4[i,:]=region_to_bbox(gt[i,:],center=False)
        new_distances[i]=_compute_distance(bboxes[i,:],gt4[i,:])
        #计算重叠率
        new_ious[i]=_compute_iou(bboxes[i,:],gt4[i,:])
        
    #what's the percentage of from in which center displacement is inferior to given threshold?(OTB metric)
    #sum(new_distances<dist_threshold):get the number of (new_distances<dist_threshold)
    precision=sum(new_distances<dist_threshold)/np.size(new_distances)*100
    
    #find above result for many thresholds,then report the AUC
    thresholds=np.linspace(0,25,n_thresholds+1)
    #get the number from the index of 1
    thresholds=thresholds[-n_thresholds:]
    #!!!reverse it so that higer values of precision goes at the beginning
    thresholds=thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i]=sum(new_distances<thresholds[i])/np.size(new_distances)
        
    #integrate over the thresholds
    #AUC（Area Under Curve）被定义为ROC曲线下的面积
    precision_auc=np.trapz(precisions_ths)
    
    #per frame averaged interseciton over union (OTB metric)
    iou=np.mean(new_ious)*100

    return l,precision,precision_auc,iou


#get the center distance of the two boxes
def _compute_distance(boxA,boxB):
    #get the center
    a=np.array((boxA[0]+boxA[2]/2,boxA[1]+boxA[3]/2))
    b=np.array((boxB[0]+boxB[2]/2,boxB[1]+boxB[3]/2))
    #范式
    dist=np.linalg.norm(a-b)
    
    assert dist>=0
    assert dist!=float('Inf')
    
    return dist


def _compute_iou(boxA,boxB):
    #determine the (x,y) -cooddinates of the intersection rectangle
    xA=max(boxA[0],boxB[0])
    yA=max(boxA[1],boxB[1])
    xB=min(boxA[0]+boxA[2],boxB[0]+boxB[2])
    yB=min(boxA[1]+boxA[3],boxB[1]+boxB[3])
    
    if xA<xB and yA<yB:
        #compute the area of intersection rectangle
        interArea=(xB-xA)*(yB-yA)
        #compute the area of both the prediction and ground-truth\
        #rectangles
        boxAArea=boxA[2]*boxA[3]
        boxBArea=boxB[2]*boxB[3]
        #compute the intersection over union by taking the intersection
        #area and dividing it by the sum of prediction + ground-truth
        #areas - the intersection area
        iou=interArea/float(boxAArea+boxBArea-interArea)
    else:
        iou=0
        
    assert iou>=0
    assert iou<=1.01
    
    return iou
    
if __name__=='__main__':
    sys.exit(main())