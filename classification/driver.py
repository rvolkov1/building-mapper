'''
Author: Dylan Blake
example for classification calling
'''

from classify import FWC_Classification
from slic import SLIC
import os
import cv2
import matplotlib.pyplot as plt
import pickle

if __name__=='__main__':
    debug = False
    load = True

    directory = 'C:\\UofM CompSci Masters\\Computer Vision 5561\\Project\\Dataset\\data\\0016\\panos'
    filename = 'floor_02_partial_room_04_pano_17.jpg'
    full_path = os.path.join(directory, filename)
    im = cv2.imread(full_path, flags=cv2.IMREAD_COLOR) #Same as IMREAD_COLOR_BGR
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    K = 1500
    if load:
        file = os.path.join('./', filename[:-4], 'centers')
        with open(file, 'rb') as file_open:
            centers = pickle.load(file_open)
        file = os.path.join('./', filename[:-4], 'labels')
        with open(file, 'rb') as file_open:
            labels = pickle.load(file_open)
        file = os.path.join('./', filename[:-4], 'segmented_img_rgb.png')
        segmented_img_rgb = cv2.imread(file)
    else:
        slic = SLIC(k=K, spatial_bias=10)
        segmented_img_rgb, centers, labels = slic.run(im)

    if debug:
        slic.save_border_img(filename[:-4], segmented_img_rgb, labels)
        file = os.path.join('./', filename[:-4], 'centers')
        with open(file, 'wb') as file_p:
            pickle.dump(centers, file_p)
        file = os.path.join('./', filename[:-4], 'labels')
        with open(file, 'wb') as file_p:
            pickle.dump(labels, file_p)
        plt.imsave(os.path.join('./', filename[:-4], 'segmented_img_rgb.png'), segmented_img_rgb)
    
    classifer = FWC_Classification(segmented_img_rgb, K, centers, labels)
    classifer.classify_fwc(im, segmented_img_rgb, D_max=9.0, FLOOR_OFFSET=3, CEILING_OFFSET=3)
    #s = 8
    #for D_max in [s*0.85, s*0.9, s*0.95, s*1.0, s*1.05, s*1.10, s*1.15, s*1.2]:
    #    im_labeled, labeled_clusters = classify_fwc(im_lab, im_avg, centers, labels, h, w, S, FLOOR_OFFSET, CEILING_OFFSET, D_max, spatial_bias)
    #    plt.imsave(os.path.join('./', filename[:-4], f'classified_D_{D_max}.png'), im_labeled)
    #im_labeled, labeled_clusters = classify_fwc(im_lab, im_avg, centers, labels, h, w, S, FLOOR_OFFSET, CEILING_OFFSET, 9.0, spatial_bias)
    #plt.imsave(os.path.join('./', filename[:-4], f'classified_D_{9.0}.png'), im_labeled)
