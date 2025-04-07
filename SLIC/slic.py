import cv2
import matplotlib.pyplot as plt 
import numpy as np

'''
Author: Dylan Blake

our hyperparams are the following

param spatial_bias : "allows us to weigh the relative importance bewteen color similarity and spatial proximity
           when m is large, spatial proximity is more important and the resulting superpixels are more
           compact. When m is small, the resulting superpixels adhere more tightly to image boundaries
           but have less regular size and shape."

           put simply, just by looking at the equation, larger m means the spatial component increasingly dominates 
           the Lab color component. we see the borders begin to increasingly resemble an SxS region.
           see line 86 in this file for the equation.
           
param k : larger k decreases cell size meaning more resultant superpixels. 
'''

'''
Input: 
    ctrs: the current centers in k means
    labels: array of labels in which i'th entry belongs to the label[i]'th cluster  

    following same pattern as k means update centers - the new center of a superpixel is the average of all its pixels in CIELAB space
'''
def update_centers(ctrs, labels, h, w):
    new_ctrs = np.zeros_like(ctrs)
    cts = np.zeros(ctrs.shape[0])
    for m in range(h):
        for n in range(w):
            label = labels[m, n]
            if label != 0:
                new_ctrs[label][:3] += im[m, n]
                new_ctrs[label][3:] += [m, n]
                cts[label] += 1
    for i in range(ctrs.shape[0]):
        if cts[i] > 0:
            new_ctrs[i] /= cts[i]
    return new_ctrs
'''
initialize k means with centers located along a grid of cells
Input: 
    im - image in Lab color space
    S - sampling interval, the cell size is S*S
    h - height of image
    w - width of image
Output: 
    k means initial centers (l, a, b, x, y)
'''
def init_centers(im, S, h, w):
    ctrs = []
    for m in range(S // 2, h, S):
        for n in range(S // 2, w, S):
            color = im[m, n] #Lab space color
            ctrs.append([color[0], color[1], color[2], n, m])
    return np.array(ctrs, dtype=np.float64)
'''
SLIC is basically just k-means
Input: 
    im - image in Lab color space - CIELAB
    k - number of clusters basically
    spatial_bias - 'compactness' coefficient
    border - display border around superpixels
    iterations - the number of times we update the k means cluster centers
'''
def SLIC(im, k=200, spatial_bias=30, border=True, iterations=10):
    h, w = im.shape[0], im.shape[1]
    S = int(np.sqrt(h * w / k))

    ctrs = init_centers(im,S,h,w)

    labels = np.zeros(shape=(h,w), dtype=np.uint32)
    distances = np.full(shape=(h,w), fill_value=np.inf)

    for _ in range(iterations):
        for idx, (l,a,b,cx,cy) in enumerate(ctrs):
            left = max(int(cx - 2*S), 0)
            right = min(int(cx + 2*S), w)
            top = max(int(cy - 2*S), 0)
            bottom = min(int(cy + 2*S), h)

            cell = im[top:bottom, left:right] #search space - compute distances to cluster center - ensure we overlap with neighboring cells with 2*S
            cell_x, cell_y = np.meshgrid(np.arange(left, right), np.arange(top, bottom))

            cdist = cell - np.array([l,a,b])
            dc = np.linalg.norm(cdist, axis=2) #axis 0 is cell row, 1 cell column, 2 is cell color
            ds = np.sqrt((cell_x - cx)**2 + (cell_y - cy)**2)

            D = np.sqrt(dc**2 + (ds / S)**2 * spatial_bias**2) #See 'SLIC Superpixels Compared to State-of-the-Art Superpixel Methods' equation (3)

            mask = D < distances[top:bottom, left:right] #pixels within the cell where D is smallest
            distances[top:bottom, left:right][mask] = D[mask]
            labels[top:bottom, left:right][mask] = idx
        ctrs = update_centers(ctrs, labels, h, w)

    segmented_img_lab = np.zeros_like(im)
    for i in range(ctrs.shape[0]):
        mask = (labels == i) #2D mask of all pixels in i'th cluster
        if np.any(mask):
            mean_color = im[mask].mean(axis=0)
            segmented_img_lab[mask] = mean_color

    output = segmented_img_lab.astype(np.uint8)

    if border:
        boundaries = np.zeros_like(labels, dtype=bool)
        for m in range(1, h - 1):
            for n in range(1, w - 1):
                if np.any(labels[m, n] != labels[m-1:m+2, n-1:n+2]):
                    boundaries[m, n] = True
        
        output = cv2.cvtColor(output, cv2.COLOR_LAB2RGB)
        output[boundaries] = [0, 0, 0] 
    else:
        output = cv2.cvtColor(output, cv2.COLOR_LAB2RGB)

    plt.imshow(output)
    plt.title(f"SLIC Superpixels k:{k} spatial_bias:{spatial_bias}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    im = cv2.imread('res/mona_lisa.jpg', flags=cv2.IMREAD_COLOR) #Same as IMREAD_COLOR_BGR
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    im_lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

    SLIC(im_lab)
    #for m in m_arr:
    #    SLIC(im_lab)
    #for k in k_arr:
    #    SLIC(im_lab, k, 10)