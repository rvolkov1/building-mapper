"""
Author: Ross Volkov
"""

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2

from feature_matching.feature_matching_utils import load_imgs_gray, show_imgs, visualize_sift, find_match, visualize_find_match, compute_F, visualize_epipolar_lines, compute_camera_pose, visualize_camera_poses

def main():
  parser = argparse.ArgumentParser(description='Extract 2D perspective views from panoramic images.')
  parser.add_argument('--base_dir', type=str, default='zind_subset',
    help='Root directory containing scene folders (default: zind_subset)')
    
  args = parser.parse_args()

def build_camera_rot(angle):
  theta = np.deg2rad(angle)
  z = np.array([np.cos(theta), np.sin(theta), 0])
  y = np.array([0, 0, 1])
  x = np.cross(y, z)
  R = np.stack([x, y, z], axis=1)
  return R

#def get_camera_instrinsics():
#  fov = 73.8
#  camera_pos_0 = np.array([2.97, -0.87])
#  camera_rot_0 = 265.97
#  camera_yaw_0 = -240.31
#  camera_dist_to_target_0 = 2.44
#
#  camera_pos_1 = np.array([3.54, -1.44])
#  camera_rot_1 = -125.91
#  camera_yaw_1 = 170.73
#  camera_dist_to_target_1 = 2.30
#
#  R_0 = build_camera_rot(camera_rot_0)
#  R_1 = build_camera_rot(camera_rot_1)
#
#  R = R_1 @ R_0.T
#  c = R_1 @ (camera_pos_0 - camera_pos_1)

def get_camera_intrinsics(w, h):
  theta_x = np.deg2rad(73.8)
  fx = w/(2*np.tan(theta_x/2))

  # using fov_h for fov_w, but this is not necessarily correct...
  fy = h/(2*np.tan(theta_x/2))

  cx = w/2
  cy = h/2

  return np.array(
    [[fx, 0,  cx],
     [0,  fy, cy],
     [0,  0,  0]])

def get_3d_pt_cloud(path):
  img_paths = [img for img in os.listdir(path) if ".jpg" in img]
  print(img_paths)
  imgs = load_imgs_gray(path, ["view_0.jpg", "view_1.jpg"])
  show_imgs(imgs)

  im1 = imgs[0]
  im2 = imgs[1]

  visualize_sift(im1)
  visualize_sift(im2)

  x1, x2 = find_match(im1, im2, dist_thr=0.8)
  visualize_find_match(im1, im2, x1, x2)

  print(x1.shape)

  #F = compute_F((x1, x2), eps=2)
  F, mask = cv2.findFundamentalMat(x1, x2,cv2.FM_LMEDS)

  x1 = x1[mask.ravel()==1]
  x2 = x2[mask.ravel()==1]
  
  print("F: ", F)
  visualize_epipolar_lines(im1, im2, (x1, x2), F)

  h, w = im1.shape
  h2, w2 = im2.shape
  K1 = get_camera_intrinsics(w, h)
  K2 = get_camera_intrinsics(w2, h2)

  Rs, Cs = compute_camera_pose(F, K1)

  visualize_camera_poses(Rs, Cs)


def all_opencv(path):
  img_paths = [img for img in os.listdir(path) if ".jpg" in img]
  print(img_paths)
  imgs = load_imgs_gray(path, ["view_0.jpg", "view_1.jpg"])
  show_imgs(imgs)

  im1 = imgs[0]
  im2 = imgs[1]  

  sift = cv2.SIFT_create()
  
  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(im1,None)
  kp2, des2 = sift.detectAndCompute(im2,None)
  
  # FLANN parameters
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks=50)
  
  flann = cv2.FlannBasedMatcher(index_params,search_params)
  matches = flann.knnMatch(des1,des2,k=2)
  
  pts1 = []
  pts2 = []

  
  # ratio test as per Lowe's paper
  for i,(m,n) in enumerate(matches):
      if m.distance < 0.7*n.distance:
          pts2.append(kp2[m.trainIdx].pt)
          pts1.append(kp1[m.queryIdx].pt)

  pts1 = np.int32(pts1)
  pts2 = np.int32(pts2)

  print("pts shapes: ", pts1.shape, pts2.shape)

  F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
  
  # We select only inlier points
  pts1 = pts1[mask.ravel()==1]
  pts2 = pts2[mask.ravel()==1]

  def drawlines(img1,img2,lines,pts1,pts2):
      ''' img1 - image on which we draw the epilines for the points in img2
          lines - corresponding epilines '''
      r,c = img1.shape
      img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
      img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
      for r,pt1,pt2 in zip(lines,pts1,pts2):
          color = tuple(np.random.randint(0,255,3).tolist())
          x0,y0 = map(int, [0, -r[2]/r[1] ])
          x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
          img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
          img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
          img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
      return img1,img2

  # Find epilines corresponding to points in right image (second image) and
  # drawing its lines on left image
  lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
  lines1 = lines1.reshape(-1,3)
  img5,img6 = drawlines(im1, im2,lines1,pts1,pts2)
  
  # Find epilines corresponding to points in left image (first image) and
  # drawing its lines on right image
  lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
  lines2 = lines2.reshape(-1,3)
  img3,img4 = drawlines(im2,im1,lines2,pts2,pts1)
  
  plt.subplot(121),plt.imshow(img5)
  plt.subplot(122),plt.imshow(img3)
  plt.show()
