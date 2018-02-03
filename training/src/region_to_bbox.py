import numpy as np

def region_to_bbox(region,center=True):
    
    n=len(region)
    assert n==4,('GT region format is invalid,should have 4 entries')
    
    if n==4:
        return _rect(region,center)
    
#the groundtruth is saved as a rect(lx,ly,w,h)
def _rect(region,center):
    
    if center:
        x=region[0]
        y=region[1]
        w=region[2]
        h=region[3]
        cx=x+w/2
        cy=y+h/2
        
        return cx,cy,w,h
    #not center,return rect
    else:
        return region
    