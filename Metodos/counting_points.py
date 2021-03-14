import scipy.ndimage as ndi
import numpy as np
import skimage.draw as skidraw
import matplotlib.pyplot as plt

def count_agreements(img, points):
    '''
    `img` is a binary image containing ground truth marks and `points` 
    are the points detected by the algorithm
    
    Parameters
    ----------
    img : numpy array
        Binary image
    points : numpy array
        Nx2 array containing the points
        
    Returns
    -------
    tp : int
        True positives
    fp : int
        False positives
    fn : int
        False negatives
    '''

    img_colors = np.tile(img[:,:,None], (1, 1, 3))
    img_colors[:,:,:] = (0,0,0) 
    img_points = np.zeros_like(img, dtype=np.uint8)
    img_points[points[:,0],points[:,1]] = 1
    #print("num_points:",np.sum(img_points))

    
    for point in points:
        rr,cc = skidraw.disk((point[0],point[1]),2)
        img_colors[rr,cc] = (255,255,255)
    
    

    img_labels, num_comp = ndi.label(img)
    slices = ndi.find_objects(img_labels)
  

   

    tp = fn = fp = 0
    for index, slice in enumerate(slices):
        if slice!=None:
            img_comp = ((img_labels[slice]==(index+1)).astype(np.uint8))
            img_comp_points = (img_points[slice]*img_comp).astype(np.uint8)
            points = list(np.nonzero(img_comp_points))
            points[0] += slice[0].start 
            points[1] += slice[1].start 
            

            num_points_in_comp = np.sum(img_comp_points)
            if num_points_in_comp==0:
                img_colors[slice] = (255,0,0)
                fn += 1
            elif num_points_in_comp == 1:
                tp +=1
                img_colors[slice] = img_colors[slice] * (0,1,0)
            else:
                img_colors[slice] = img_colors[slice] * (0,0,1)
                tp += 1
                fp += num_points_in_comp-1
                
    fp += np.sum(np.logical_not(img>0)*img_points)
    plt.imsave("TpFpFn_Colors",img_colors,format="pdf")
    plt.imshow(img_colors)
    plt.show()
    
    return tp, fp, fn
        
def draw_points(img, points):
    
    img_points = np.tile(img[:,:,None], (1, 1, 3))
    img_points[points[:,0], points[:,1]] = (255, 0, 0)
    plt.imshow(img_points)
    plt.show()


'''
img = plt.imread('test.tiff')
points = np.array([(16, 21), (14, 65), (16, 60), (20, 89), (37, 82), (39, 78), (43, 11), (45, 45),
                  (52, 10), (59, 36), (59, 66), (70, 19), (74, 20), (74, 25), (75, 17), (78, 47)])

draw_points(img, points)
tp, fp, fn = count_agreements(img, points)
print(f'tp = {tp}, fp = {fp}, fn = {fn}')
'''