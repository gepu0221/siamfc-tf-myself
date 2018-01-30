import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def show_frame(frame,bbox,fig_n):
    #plt.figure(index):创建一个序号为index的图，可以通过序号调用
    fig=plt.figure(fig_n)
    #fig.add_subplot(349)-----349：将画布分割成3行4列，图像画在从左到右从上到下的第9块
    ax=fig.add_subplot(111)
    r=patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='g',fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    #打开交互模式
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.clf()

def save_frame(frame,bbox,fig_n,frame_name):
    fig=plt.figure(fig_n)
    ax=fig.add_subplot(111)
    #r=patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='g',fill=False)
    #ax.add_patch(r)
    
    #draw the ellipse
    c_x=bbox[0]+bbox[2]/2
    c_y=bbox[1]+bbox[3]/2
    ell=Ellipse(xy=(c_x,c_y),width=bbox[2],height=bbox[3],angle=0,facecolor='none',edgecolor='g')
    ax.add_patch(ell)
    
    #x,y=c_x,c_y
    #ax.plot(x,y,'ro')
    
    ax.imshow(np.uint8(frame))
    fig.savefig(frame_name)
    plt.clf()
    
def show_crops(crops,fig_n):
    fig=plt.figure(fig_n)
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)
    ax1.imshow(np.uint8(crops[0,:,:,:]))
    ax2.imshow(np.unit8(crops[1,:,:,:]))
    ax3.imshow(np.unit8(crops[2,:,:,:]))
    plt.ion()
    plt.show()
    plt.pause(0.001)
    
    
def show_scores(scores,fig_n):
    fig=plt.figure(fig_n)
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_supplot(133)
    ax1.imshow(scores[0,:,:],interpolation='none',cmap='hot')
    ax2.imshow(scores[1,:,:],interpolation='none',cmap='hot')
    ax3.imshow(scores[2,:,:],interpolation='none',cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.001)