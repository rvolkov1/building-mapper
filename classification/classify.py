'''
Author: Dylan Blake
'''
import cv2
import numpy as np
import heapq
from enum import IntEnum
import os
import sys
from utils import transform_util, spherical_to_perspective, perspective_to_spherical
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_utils import equirectangular_to_perspective
from homography import homography_RANSAC
from vanishing_points import detect_lines, get_vanishing_points

class Label(IntEnum):
    UNKNOWN=0
    FLOOR=1
    WALL=2
    CEILING=3

class FWC_Classification():
    def __init__(self, segmented_img_rgb, k, centers, labels):
        '''
        input:
        centers: cluster center output from SLIC
        labels: cluster labels output from SLIC
        h: image height
        w: image width
        k: target number of clusters
        spatial_bias: preference towards spatial distance over LAB color distance
        '''
        h, w = segmented_img_rgb.shape[:2]
        self.centers = centers
        self.labels = labels
        self.S = int(np.sqrt(h * w / k))
        self.cl_h, self.cl_w = int(h / self.S + 0.5), int(w / self.S + 0.5) #cluster height, cluster width
        self.sp_labels_fwc = np.zeros(shape=(self.cl_h, self.cl_w), dtype=Label) #add 0.5 to round to the nearest int
        self.p_labels_fwc = np.zeros_like(labels)
        assert  self.cl_h*self.cl_w == centers.shape[0]
        
    def get_neighbors(self, m, n):
        '''
        returns neighbors within the 3x3 surrounding region of a pixel at (m, n)
        '''
        neighbors = []
        for i in range(m-1, m+2):
            if i < self.cl_h and i > 0:  
                for j in range(n-1, n+2):
                    if j < self.cl_w and j > 0:
                        if i != m or j != n:
                            neighbors.append((i,j))
        return np.array(neighbors)
    
    def get_labeled_image(self, segmented_img_rgb):
        '''
        input: super-pixel segmented image where each sp has its average color
        returns super-pixel labels overlayed onto the color averages
        '''
        im_labeled = np.copy(segmented_img_rgb)
        flat_sp_labels = self.sp_labels_fwc.flatten()
        for i in range(self.centers.shape[0]):
            mask = (self.labels == i) #2D mask of all pixels in i'th cluster
            if np.any(mask):
                color = None
                if flat_sp_labels[i] == Label.FLOOR:
                    color = np.array([255,0,0], dtype=np.float64)
                elif flat_sp_labels[i] == Label.WALL:
                    color = np.array([0,255,0], dtype=np.float64)
                elif flat_sp_labels[i] == Label.CEILING:
                    color = np.array([0,0,255], dtype=np.float64)
                if flat_sp_labels[i] != Label.UNKNOWN:
                    im_labeled[mask] = 0.6*segmented_img_rgb[mask] + 0.4*color
        im_labeled = im_labeled.astype(np.uint8)
        return im_labeled    

    def update_pixel_wise_labels(self):
        '''
        update pixel-wise labels using super-pixel-wise labels
        '''
        flat_sp_labels = self.sp_labels_fwc.flatten()
        for i in range(self.centers.shape[0]):
            mask = (self.labels == i) #2D mask of all pixels in i'th cluster the labels used here are not fwc, but labels identifying distinct sp's
            if np.any(mask):
                self.p_labels_fwc[mask] = flat_sp_labels[i]
    def update_sp_pixel_ownership_map(self):
        flat_sp_labels = self.sp_labels_fwc.flatten()
        for i in range(self.centers.shape[0]):
            mask = (self.labels == i) #2D mask of all pixels in i'th cluster the labels used here are not fwc, but labels identifying distinct sp's
            if np.any(mask):
                self.p_labels_fwc[mask] = flat_sp_labels[i]
    def pixel_is_in_sp(self, sm, sn, pm, pn):
        i = sm*self.cl_w+sn
        mask = (self.labels == i) #2D mask of all pixels in i'th cluster the labels used here are not fwc, but labels identifying distinct sp's
        return mask[pm, pn]
    def conservative_propagation(self, CEILING_OFFSET, FLOOR_OFFSET, D_max):
        '''
        conservative propagation - iteratively propagate the labeling of each super-pixel to its neighbors
        this takes a breadth-first-search approach, but only propagates if the distance measure D is less than D_max between neighboring super-pixels
        
        input:
        CEILING_OFFSET: initial row offset from the 0'th row for labeling super-pixels as CEILING
        FLOOR_OFFSET: initial row offset from the cl_h'th row for labeling super-pixels as FLOOR
        D_max: hyperparameter for propagation
        '''
        #various maxima and minima we'd like to keep track of
        self.min_ceiling, self.min_ceiling_row = None, CEILING_OFFSET
        self.max_floor, self.max_floor_row = None, self.cl_h-FLOOR_OFFSET
        self.min_wall, self.min_wall_row = None, self.cl_h
        self.max_wall, self.max_wall_row = None, 0

        #label top-most row of super-pixels as ceiling
        self.sp_labels_fwc[CEILING_OFFSET,:] = Label.CEILING 

        #label bottom-most row of super-pixels as floor
        self.sp_labels_fwc[-1-FLOOR_OFFSET,:] = Label.FLOOR 

        #label middle row of super-pixels as wall
        mid = int(self.sp_labels_fwc.shape[0] / 2)
        self.sp_labels_fwc[mid] = Label.WALL 
        
        #to init the propagation, push pre-initialized labels to the heap with zero cost
        initial_labeled_rows = [CEILING_OFFSET, mid, self.cl_h-1-FLOOR_OFFSET]
        pq = [] #priority queue
        for row in initial_labeled_rows:
            for n in range(self.cl_w):
                heapq.heappush(pq, (0, (row, n)))

        while len(pq) > 0:
            current_superpixel = heapq.heappop(pq)
            cm, cn = current_superpixel[1]
            assert self.sp_labels_fwc[cm, cn] != Label.UNKNOWN
            cl, ca, cb, cx, cy = self.centers[cm*self.cl_w + cn]
            neighbors = self.get_neighbors(cm, cn)
            for neighbor in neighbors:
                nm, nn = neighbor
                nl, na, nb, nx, ny = self.centers[nm*self.cl_w+nn]
                if (self.sp_labels_fwc[nm, nn] == Label.UNKNOWN):
                    cdist = np.array([nl, na, nb]) - np.array([cl,ca,cb])
                    dc = np.linalg.norm(cdist)
                    ds = np.sqrt((nx - cx)**2 + (ny - cy)**2)
                    D = np.sqrt(dc**2 + (ds / self.S)**2) #See 'SLIC Superpixels Compared to State-of-the-Art Superpixel Methods' equation (3), remove spatial bias, it can be expressed by our D_max hyperparameter
                    if D < D_max:
                        self.sp_labels_fwc[nm,nn] = self.sp_labels_fwc[cm, cn] #set the label to the label of the current
                        heapq.heappush(pq, (D, (nm, nn)))
                        if self.sp_labels_fwc[nm, nn] == Label.FLOOR:
                            if nm > self.max_floor_row:
                                self.max_floor = self.centers[nm*self.cl_w+nn][4]
                                self.max_floor_row = nm
                        elif self.sp_labels_fwc[nm, nn] == Label.CEILING:
                            if nm < self.min_ceiling_row:
                                self.min_ceiling = self.centers[nm*self.cl_w+nn][4]
                                self.min_ceiling_row = nm
                        elif self.sp_labels_fwc[nm, nn] == Label.WALL:
                            if nm < self.min_wall_row:
                                self.min_wall = self.centers[nm*self.cl_w+nn][4]
                                self.min_wall_row = nm
                            if nm > self.max_wall_row:
                                self.max_wall = self.centers[nm*self.cl_w+nn][4]
                                self.max_wall_row = nm
    def enforce_label_order(self):
        '''
        Cabral step 4 Enforce label order: 
        Every pixel above (resp. below) the top-most ceiling (resp. bottom-most floor) pixel is also labeled as ceiling (resp. floor)
        Every pixel between the top-most and bottom-most wall pixels is assigned a wall label. 
        We also exploit the homography mapping: For each pixel w/ a floor label, we label the corresponding pixel through the homography as ceiling, if it does not already have a label
        '''
        for m in range(self.cl_h):
            for n in range(self.cl_w):
                if self.sp_labels_fwc[m, n] == Label.UNKNOWN:
                    _, _, _, _, y = self.centers[m*self.cl_w+n]
                    if self.max_floor is not None and y > self.max_floor:
                        self.sp_labels_fwc[m, n] = Label.FLOOR
                    if self.min_ceiling is not None and y < self.min_ceiling:
                        self.sp_labels_fwc[m, n] = Label.CEILING
                    if self.min_wall is not None and y > self.min_wall + self.S and self.max_wall is not None and y < self.max_wall - self.S:
                        self.sp_labels_fwc[m, n] = Label.WALL
    
    def classify_fwc(self, spherical_im_rgb, spherical_segmented_img_rgb, D_max, FLOOR_OFFSET, CEILING_OFFSET, visualize=False):
        '''
            classify floor-wall-ceiling
            input:
                spherical_im_rgb: equirectangular image in rgb color space
                spherical_segmented_img_rgb: equirectangular image output from SLIC 
                D_max: threshold for label propagation
                FLOOR_OFFSET: offsets initial floor labeling
                CEILING_OFFSET: offsets initial ceiling labeling
        '''
        self.conservative_propagation(CEILING_OFFSET, FLOOR_OFFSET, D_max)
        #current_labeling = self.get_labeled_image(spherical_segmented_img_rgb)
        #cv2.imshow("cp_labeling", current_labeling)
        self.enforce_label_order()
        current_labeling = self.get_labeled_image(spherical_segmented_img_rgb)
        cv2.imshow("result", current_labeling)
        plt.imsave(os.path.join('./', 'floor_02_partial_room_04_pano_17', 'segmented_img_rgb.png'), current_labeling)
        #cv2.imshow("enforce_labeling", current_labeling)
        self.update_pixel_wise_labels()
        h, w = spherical_im_rgb.shape[:2]
        #convert sp labels to global image space

        fov = 120
        views = [0, 120, 240]
        #we partition the panorama into 512x512 slices  
        for view in views:
            perspective_labels = equirectangular_to_perspective(self.p_labels_fwc, fov, view).astype(np.uint8)
            perspective_segmented_im = equirectangular_to_perspective(spherical_segmented_img_rgb, fov, view).astype(np.uint8)
            perspective_im = equirectangular_to_perspective(spherical_im_rgb, fov, view).astype(np.uint8)
            #labeling_im = equirectangular_to_perspective(current_labeling, fov, view).astype(np.uint8)
            #convert cluster centers to perspective - this is a flat array where the i'th entry corresponds to sp_labels_fwc[i//cl_h, i%cl_h], so we don't need to convert back since we have a direct mapping to a data structure in spherical space
            perspective_centers_x, perspective_centers_y = spherical_to_perspective(self.centers[:,3], self.centers[:,4], w, h,fov,view) #[x, y]
            if visualize:
                cv2.imshow("perspective_segmented_image", perspective_segmented_im)
                cv2.imshow("perspective original image", perspective_im)
                #cv2.imshow("perspective labeling", labeling_im)
            im_canny, lines = detect_lines(perspective_im, visualize)
            cv2.imshow("edge", im_canny)
            vl, vr = get_vanishing_points(lines, visualize, perspective_im)
            #horizon_line = np.cross(vl, vr)
            #horizon_line = horizon_line / horizon_line[2] 
            horizon_y = int((vl[1] + vr[1]) / 2) #vl and vr are expected to be approximately at the same y
            H = homography_RANSAC(perspective_im, perspective_labels, horizon_y)
            _, sp_horizon_y = perspective_to_spherical(0, horizon_y, 512, 512, fov, view, 0, w, h)
            sp_horizon_y = int(sp_horizon_y / self.S + 0.5)
            #iterate through all of the superpixels above the sp_horizon_y
            for m in range(sp_horizon_y):
                for n in range(self.cl_w):
                    if self.sp_labels_fwc[m,n] == Label.CEILING or self.sp_labels_fwc[m,n] == Label.UNKNOWN:
                        cx, cy = perspective_centers_x[m*self.cl_w+n], perspective_centers_y[m*self.cl_w+n]
                        if not np.isnan(cx) and not np.isnan(cy): #check that the center is in the current view
                            # apply the homography to get a floor superpixel
                            fx, fy = transform_util(H, [[cx, cy]])[0]
                            '''
                            fig, ax = plt.subplots()
                            ax.imshow(perspective_segmented_im)
                            ax.scatter([cx], [cy], color='blue', label='Source Points', s=100, edgecolors='k')
                            ax.scatter([fx], [fy], color='red', label='Destination Points', s=100, edgecolors='k')
                            ax.plot([cx, fx], [cy, fy], 'k--', lw=1)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            plt.title('Ceiling to Floor Point Correspondences')
                            plt.legend()
                            plt.grid(True)
                            plt.tight_layout()
                            plt.show()
                            '''
                            # convert to spherical coordinates
                            sphx, sphy = perspective_to_spherical(fx, fy, 512, 512, fov, view, 0, w, h)
                            '''
                            fig, ax = plt.subplots()
                            ax.imshow(spherical_segmented_img_rgb)
                            ax.scatter([sphx], [sphy], color='pink', label='', s=100, edgecolors='k')
                            plt.show()
                            '''
                            #convert back to super-pixel coordinate
                            sp_m, sp_n = None, None
                            sm, sn = int(sphy / self.S + 0.5), int(sphx / self.S + 0.5)
                            if self.pixel_is_in_sp(sm, sn, sphy, sphx):
                                sp_m, sp_n = sm, sn
                            candidates = self.get_neighbors(sm, sn)
                            for cand in candidates:
                                candm, candn = cand
                                if self.pixel_is_in_sp(candm, candn, sphy, sphx):
                                    sp_m, sp_n = candm, candn
                            #set superpixel to a floor, also if this is valid, then the original sp is a ceiling
                            #assert sn < self.cl_w
                            if sp_m != None and sp_m < self.cl_h and sp_m > sp_horizon_y and sp_n < self.cl_w:
                                self.sp_labels_fwc[sp_m, sp_n] = Label.FLOOR
            
            current_labeling = self.get_labeled_image(spherical_segmented_img_rgb)
            cv2.imshow(f"view {view}", current_labeling)
        self.enforce_label_order()
        current_labeling = self.get_labeled_image(spherical_segmented_img_rgb)
        cv2.imshow("result", current_labeling)