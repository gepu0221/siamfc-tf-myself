import numpy as np
import math

#LabelMapSize:w*h(w=h)
def create_label(LabelMapSize,dPos):
    #use the "balance" method(Postive and Negative each half 0.5)
    logloss_label=create_logisticloss_label(LabelMapSize,dPos)
    #label with weight normalization
    label_weight=np.ones(logloss_label.shape)
    #count the sum of Positive labels
    Psum=np.sum(logloss_label==1)
    #count the sum of Negative labels
    Nsum=np.sum(logloss_label==-1)
    
    l_sz=LabelMapSize[0]
    
    weight_p=1/Psum*0.5
    weight_n=1/Nsum*0.5
    
    for i in range(0,l_sz):
        for j in range(0,l_sz):
            if(logloss_label[i][j]==1):
                label_weight[i][j]=weight_p
            elif(logloss_label[i][j]==-1):
                label_weight[i][j]=weight_n
        
    return label_weight

def create_logisticloss_label(LabelMapSize,dPos):
    l_sz=LabelMapSize[0]
    #math.ceil()向上取整
    #round():四舍五入取整
    pos_x,pos_y=int(l_sz/2),int(l_sz/2)
    logloss_label=np.zeros([l_sz,l_sz])
    
    for i in range(0,l_sz):
        for j in range(0,l_sz):
            dist=(pos_x-i)**2+(pos_y-j)**2
            if dist<=dPos:
                logloss_label[i][j]=1
            else:
                logloss_label[i][j]=-1
                
    return logloss_label