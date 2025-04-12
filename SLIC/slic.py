import cv2
import matplotlib.pyplot as plt 
import numpy as np
import heapq
from enum import IntEnum
import pickle
import os
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
                new_ctrs[label][3:] += [n, m]
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
SLIC is basically just k nearest neighbors
Input: 
    im - image in Lab color space - CIELAB
    k - number of clusters basically
    spatial_bias - 'compactness' coefficient
    border - display border around superpixels
    iterations - the number of times we update the k means cluster centers
'''
def SLIC(im, S, spatial_bias=10, iterations=10):
    '''
    output = cv2.cvtColor(im, cv2.COLOR_LAB2RGB)
    for center in ctrs:
        n, m = int(center[3]), int(center[4])
        cv2.drawMarker(output, (n,m), color=(255,255,0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1, line_type=cv2.LINE_AA)
    plt.imshow(output)
    plt.show()
    '''
    h, w = im.shape[:2]
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

    return (output, ctrs, labels)

'''
    classify floor-wall-ceiling
    param centers - cluster centers from SLIC
    param labels - HxW image where labels[m, n] = i corresponds to the i'th cluster center
'''
def classify_fwc(im, im_avg, centers, labels, im_h, im_w, S, FLOOR_OFFSET, D_max = 0.85, spatial_bias=10, ):
    '''
    im_avg = np.zeros_like(im)
    for i in range(centers.shape[0]):
        mask = (labels == i) #2D mask of all pixels in i'th cluster
        if np.any(mask):
            mean_color = im[mask].mean(axis=0)
            #mean_color = centers[i][:3]
            im_avg[mask] = mean_color
    im_avg = cv2.cvtColor(im_avg, cv2.COLOR_LAB2RGB)
    plt.imshow(im_avg)
    plt.show()
    im_avg = im_avg.astype(np.float64)
    '''

    class Label(IntEnum):
        UNKNOWN=0
        FLOOR=1
        WALL=2
        CEILING=3
        EXPLORED=4 #we've explored, but conservative propagation failed - D was greater than D_max

    labeled_clusters = np.zeros(shape=(int(im_h / S + 0.5), int(im_w / S + 0.5)), dtype=Label) #add 0.5 to round to the nearest int
    cl_h, cl_w = labeled_clusters.shape[:2] #cluster height, cluster width
    assert  cl_h*cl_w == centers.shape[0]

    #label top-most row of super-pixels as ceiling
    labeled_clusters[0,:] = Label.CEILING
    #label bottom-most row of super-pixels as floor
    labeled_clusters[-FLOOR_OFFSET,:] = Label.FLOOR #bottom row tends to capture the panoramic camera itself so offset set this by another row
    #label middle row of super-pixels as wall
    mid = int(labeled_clusters.shape[0] / 2)
    labeled_clusters[mid] = Label.WALL
    #conservative propagation - iteratively propagate the labeling of each super-pixel to its neighbors
    def get_neighbors(m, n):
        neighbors = []
        for i in range(m-1, m+2):
            if i < cl_h and i > 0:  
                for j in range(n-1, n+2):
                    if j < cl_w and j > 0:
                        if i != m or j != n:
                            neighbors.append((i,j))
        return np.array(neighbors)

    #to init the propagation, get unlabeled superpixels adjacent to the already labeled rows, this is the seed for the heap 
    initial_labeled_rows = [0, mid, cl_h-FLOOR_OFFSET]
    pq = [] #priority queue
    for row in initial_labeled_rows:
        for n in range(cl_w):
            heapq.heappush(pq, (0, (row, n)))
    #iter = 0
    while len(pq) > 0:
        current_superpixel = heapq.heappop(pq)
        cm, cn = current_superpixel[1]
        assert labeled_clusters[cm, cn] != Label.UNKNOWN
        cl, ca, cb, cx, cy = centers[cm*cl_w + cn]
        neighbors = get_neighbors(cm, cn)
        for neighbor in neighbors:
            nm, nn = neighbor
            nl, na, nb, nx, ny = centers[nm*cl_w+nn]
            if (labeled_clusters[nm, nn] == Label.UNKNOWN):
                cdist = np.array([nl, na, nb]) - np.array([cl,ca,cb])
                dc = np.linalg.norm(cdist)
                ds = np.sqrt((nx - cx)**2 + (ny - cy)**2)
                #D = np.sqrt(dc**2 + (ds / S)**2 * spatial_bias**2) #See 'SLIC Superpixels Compared to State-of-the-Art Superpixel Methods' equation (3)
                D = np.sqrt(dc**2 + (ds / S)**2) #See 'SLIC Superpixels Compared to State-of-the-Art Superpixel Methods' equation (3)
                if D < D_max:
                    #iter += 1
                    labeled_clusters[nm,nn] = labeled_clusters[cm, cn] #set the label to the label of the current 
                    heapq.heappush(pq, (D, (nm, nn)))
                    #####view the current labeling
                    '''
                    if iter % 30 == 0:
                        flat_labels = labeled_clusters.flatten()
                        im_labeled = np.zeros_like(im_avg)
                        for i in range(centers.shape[0]):
                            mask = (labels == i) #2D mask of all pixels in i'th cluster
                            if np.any(mask):
                                color = None
                                if flat_labels[i] == Label.FLOOR:
                                    color = np.array([255,0,0], dtype=np.float64)
                                elif flat_labels[i] == Label.WALL: 
                                    color = np.array([0,255,0], dtype=np.float64)
                                elif flat_labels[i] == Label.CEILING:
                                    color = np.array([0,0,255], dtype=np.float64)
                                if flat_labels[i] == Label.UNKNOWN:
                                    im_labeled[mask] = im_avg[mask]
                                else:   
                                    im_labeled[mask] = 0.6*im_avg[mask] + 0.4*color
                        im_labeled = im_labeled.astype(np.uint8)
                        plt.imshow(im_labeled)
                        plt.show()
                    #####
                    '''
    flat_labels = labeled_clusters.flatten()
    im_labeled = np.zeros_like(im_avg)
    for i in range(centers.shape[0]):
        mask = (labels == i) #2D mask of all pixels in i'th cluster
        if np.any(mask):
            color = None
            if flat_labels[i] == Label.FLOOR:
                color = np.array([255,0,0], dtype=np.float64)
            elif flat_labels[i] == Label.WALL:
                color = np.array([0,255,0], dtype=np.float64)
            elif flat_labels[i] == Label.CEILING:
                color = np.array([0,0,255], dtype=np.float64)
            if flat_labels[i] == Label.UNKNOWN:
                im_labeled[mask] = im_avg[mask]
            else:   
                im_labeled[mask] = 0.6*im_avg[mask] + 0.4*color
    im_labeled = im_labeled.astype(np.uint8)
    return (im_labeled, labeled_clusters.reshape((cl_h * cl_w, 1))) #parallel to 'centers' array

if __name__ == '__main__':
    #filename = 'C:\\UofM CompSci Masters\\Computer Vision 5561\\Project\\Dataset\\data\\0000\\panos\\floor_01_partial_room_01_pano_14.jpg'
    directory = 'C:\\UofM CompSci Masters\\Computer Vision 5561\\Project\\Dataset\\data\\0006\\panos'
    filename = 'floor_01_partial_room_05_pano_42.jpg'
    full_path = os.path.join(directory, filename)
    im = cv2.imread(full_path, flags=cv2.IMREAD_COLOR) #Same as IMREAD_COLOR_BGR
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    im_lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

    output_directory = os.path.join('./', filename[:-4])
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    '''algorithm parameters. adjust as desired'''
    h, w = im.shape[:2]
    k = 1200
    spatial_bias = 10
    S = int(np.sqrt(h * w / k))
    serialized = True #set true if you've already run as False, and the 'centers' and 'labels' files exist with the results
    visualize_borders=True
    visualize_centers=True
    '''#####################################'''

    centers = None
    labels = None
    if serialized:
        file = os.path.join('./', filename[:-4], 'centers')
        with open(file, 'rb') as file_open:
            centers = pickle.load(file_open)
        file = os.path.join('./', filename[:-4], 'labels')
        with open(file, 'rb') as file_open:
            labels = pickle.load(file_open)
    else:
        im_lab_slic, centers, labels = SLIC(im_lab, S, spatial_bias)
        im_rgb_slic = cv2.cvtColor(im_lab_slic, cv2.COLOR_LAB2RGB)

        if visualize_borders:
            im_rgb_borders = im_rgb_slic
            boundaries = np.zeros_like(labels, dtype=bool)
            for m in range(1, h - 1):
                for n in range(1, w - 1):
                    if np.any(labels[m, n] != labels[m-1:m+2, n-1:n+2]):
                        boundaries[m, n] = True
            
            im_rgb_borders[boundaries] = [50, 50, 50] 
            plt.imsave(os.path.join('./', filename[:-4], 'borders.png'), im_rgb_borders)
        if visualize_centers: 
            im_rgb_centers = im_rgb_slic
            for center in centers:
                n, m = int(center[3]), int(center[4])
                cv2.drawMarker(im_rgb_centers, (n,m), color=(255,255,0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1, line_type=cv2.LINE_AA)
            plt.imsave(os.path.join('./', filename[:-4], 'centers.png'), im_rgb_centers)

        file = os.path.join('./', filename[:-4], 'centers')
        with open(file, 'wb') as file_p:
            pickle.dump(centers, file_p)
        file = os.path.join('./', filename[:-4], 'labels')
        with open(file, 'wb') as file_p:
            pickle.dump(labels, file_p)

    im_avg = np.zeros_like(im)
    for i in range(centers.shape[0]):
        mask = (labels == i) #2D mask of all pixels in i'th cluster
        if np.any(mask):
            mean_color = im_lab[mask].mean(axis=0)
            im_avg[mask] = mean_color
    im_avg = cv2.cvtColor(im_avg, cv2.COLOR_LAB2RGB)

    plt.imsave(os.path.join('./', filename[:-4], 'output_image.png'), im_avg)

    im_rgb_borders = im_avg
    boundaries = np.zeros_like(labels, dtype=bool)
    for m in range(1, h - 1):
        for n in range(1, w - 1):
            if np.any(labels[m, n] != labels[m-1:m+2, n-1:n+2]):
                boundaries[m, n] = True
    
    im_rgb_borders[boundaries] = [50, 50, 50] 
    plt.imsave(os.path.join('./', filename[:-4], 'borders.png'), im_rgb_borders)

    im_avg = im_avg.astype(np.float64)
    #Pintore uses s = 1, but  we don't get good results with that.
    s = 8
    FLOOR_OFFSET = 3
    for D_max in [s*0.85, s*0.9, s*0.95, s*1.0, s*1.05, s*1.10, s*1.15, s*1.2]:
        im_labeled, labeled_clusters = classify_fwc(im_lab, im_avg, centers, labels, h, w, S, FLOOR_OFFSET, D_max, spatial_bias)
        plt.imsave(os.path.join('./', filename[:-4], f'classified_D_{D_max}.png'), im_labeled)