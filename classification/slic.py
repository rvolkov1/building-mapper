'''
Author: Dylan Blake
'''
import cv2
import matplotlib.pyplot as plt 
import numpy as np
import os

class SLIC():
    def __init__(self, k, spatial_bias):
        '''
        input:
        h: image height
        w: image width
        k: target number of clusters
        spatial_bias: preference towards spatial distance over LAB color distance
        '''
        self.k = k    
        self.spatial_bias = spatial_bias  
        

    def update_centers(self, im_lab, ctrs, labels, h, w):
        '''
        Input: 
            ctrs: the current centers in k means
            labels: array of labels in which i'th entry belongs to the label[i]'th cluster  
            following same pattern as k means update centers - the new center of a superpixel is the average of all its pixels in CIELAB space
        '''
        new_ctrs = np.zeros_like(ctrs)
        cts = np.zeros(ctrs.shape[0])
        for m in range(h):
            for n in range(w):
                label = labels[m, n]
                if label != 0:
                    new_ctrs[label][:3] += im_lab[m, n]
                    new_ctrs[label][3:] += [n, m]
                    cts[label] += 1
        for i in range(ctrs.shape[0]):
            if cts[i] > 0:
                new_ctrs[i] /= cts[i]
        return new_ctrs

    def init_centers(self, im, S, h, w):
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
        ctrs = []
        for m in range(S // 2, h, S):
            for n in range(S // 2, w, S):
                color = im[m, n] #Lab space color
                ctrs.append([color[0], color[1], color[2], n, m])
        return np.array(ctrs, dtype=np.float64)

    def run(self, im_rgb, iterations=10):
        '''
        interface to run the SLIC algorithm
        SLIC is basically just k means clustering
        Input: 
            im - image in RGB
            S step size for initial cluster centers
            spatial_bias - 'compactness' coefficient
            iterations - the number of times we update the k means cluster centers
        Output:
            segmented_img - segmented image where each cluster is represented by its mean RGB color
            ctrs - flat array of cluster center positions
            labels - flat array of distinct labels for each cluster
        '''
        im_lab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB) #convert to LAB color space - CIELAB
        h, w = im_lab.shape[:2]
        S = int(np.sqrt(h * w / self.k))  
        ctrs = self.init_centers(im_lab,S,h,w)
        labels = np.zeros(shape=(h,w), dtype=np.uint32)
        distances = np.full(shape=(h,w), fill_value=np.inf)

        for _ in range(iterations):
            for idx, (l,a,b,cx,cy) in enumerate(ctrs):
                left = max(int(cx - 2*S), 0)
                right = min(int(cx + 2*S), w)
                top = max(int(cy - 2*S), 0)
                bottom = min(int(cy + 2*S), h)

                cell = im_lab[top:bottom, left:right] #search space - compute distances to cluster center - ensure we overlap with neighboring cells with 2*S
                cell_x, cell_y = np.meshgrid(np.arange(left, right), np.arange(top, bottom))

                cdist = cell - np.array([l,a,b])
                dc = np.linalg.norm(cdist, axis=2) #axis 0 is cell row, 1 cell column, 2 is cell color
                ds = np.sqrt((cell_x - cx)**2 + (cell_y - cy)**2)

                D = np.sqrt(dc**2 + (ds / S)**2 * self.spatial_bias**2) #See 'SLIC Superpixels Compared to State-of-the-Art Superpixel Methods' equation (3)

                mask = D < distances[top:bottom, left:right] #pixels within the cell where D is smallest
                distances[top:bottom, left:right][mask] = D[mask]
                labels[top:bottom, left:right][mask] = idx
            ctrs = self.update_centers(im_lab, ctrs, labels, h, w)

        segmented_img = np.zeros_like(im_lab)
        for i in range(ctrs.shape[0]):
            mask = (labels == i) #2D mask of all pixels in i'th cluster
            if np.any(mask):
                mean_color = im_lab[mask].mean(axis=0)
                segmented_img[mask] = mean_color

        segmented_img = segmented_img.astype(np.uint8)
        segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_LAB2RGB)
        return (segmented_img, ctrs, labels)
    
    def save_border_img(self, directory, segmented_im, labels):
        h,w = labels.shape[:2]
        im_rgb_borders = segmented_im
        boundaries = np.zeros_like(labels, dtype=bool)
        for m in range(1, h - 1):
            for n in range(1, w - 1):
                if np.any(labels[m, n] != labels[m-1:m+2, n-1:n+2]):
                    boundaries[m, n] = True
        im_rgb_borders[boundaries] = [50, 50, 50] 
        plt.imsave(os.path.join('./', directory, 'borders.png'), im_rgb_borders)
        
    def save_centers_img(self, directory, segmented_im, centers):
        segmented_im_centers = segmented_im
        for center in centers:
            n, m = int(center[3]), int(center[4])
            cv2.drawMarker(segmented_im_centers, (n,m), color=(255,255,0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1, line_type=cv2.LINE_AA)
        plt.imsave(os.path.join('./', directory, 'centers.png'), segmented_im_centers)

if __name__ == '__main__':
    directory = 'C:\\UofM CompSci Masters\\Computer Vision 5561\\Project\\Dataset\\data\\0016\\panos'
    filename = 'floor_02_partial_room_04_pano_17.jpg'
    full_path = os.path.join(directory, filename)
    im = cv2.imread(full_path, flags=cv2.IMREAD_COLOR) #Same as IMREAD_COLOR_BGR
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    
    #im = equirectangular_to_perspective(im, 120, 0, 0, 960, 480)
    im_lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

    output_directory = os.path.join('./', filename[:-4])
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    '''algorithm parameters. adjust as desired'''
    h, w = im.shape[:2]
    k = 1500
    spatial_bias = 10
    S = int(np.sqrt(h * w / k))
    serialized = True  #set true if you've already run as False, and the 'centers' and 'labels' files exist with the results
    visualize_borders=False
    visualize_centers=False
    s = 8
    FLOOR_OFFSET = 3
    CEILING_OFFSET = 3
    '''#####################################'''




   


   
   