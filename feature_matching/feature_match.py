"""
Author: Ross Volkov
"""

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2

from feature_matching.feature_matching_utils import load_imgs_gray, show_imgs, visualize_sift, find_match, visualize_find_match, compute_F, visualize_epipolar_lines

def main():
  parser = argparse.ArgumentParser(description='Extract 2D perspective views from panoramic images.')
  parser.add_argument('--base_dir', type=str, default='zind_subset',
    help='Root directory containing scene folders (default: zind_subset)')
    
  args = parser.parse_args()

def get_3d_pt_cloud(path):
  img_paths = [img for img in os.listdir(path) if ".jpg" in img]
  imgs = load_imgs_gray(path, img_paths)
  show_imgs(imgs)

  im1 = imgs[0]
  im2 = imgs[2]

  visualize_sift(im1)
  visualize_sift(im2)


  x1, x2 = find_match(im1, im2, dist_thr=0.8)
  visualize_find_match(im1, im2, x1, x2)

  F = compute_F((x1, x2))
  print("F: ", F)
  visualize_epipolar_lines(im2, im2, (x1, x2), F)