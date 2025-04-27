"""
Author: Ross Volkov
"""

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2

from feature_matching.feature_matching_utils import load_imgs_gray, show_imgs, visualize_sift, find_match, visualize_find_match, compute_F, visualize_epipolar_lines, compute_camera_pose, visualize_camera_poses, triangulation, visualize_camera_poses_with_pts, disambiguate_pose, visualize_camera_pose_with_pts, my_warp_perspective, visualize_img_pair

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
     [0,  0,  1]])

def compute_rectification(K, R, C):
  """
  Compute the rectification homographies for both left and right images.

  Args:
  - K (np.ndarray): Intrinsic camera matrix (3x3).
  - R (np.ndarray): Rotation matrix (3x3) of the second camera.
  - C (np.ndarray): Camera center of the second camera (3x1).
  
  Returns:
  - H1 (np.ndarray): Homography for the left image (3x3).
  - H2 (np.ndarray): Homography for the right image (3x3).
  """

  # find rectification, rotation matrix R_rect s.t. x-axis alings with the baseline

  # camera one
  #P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
  #P2 = K @ R @ np.hstack([np.eye(3), -C])

  rx = C / np.linalg.norm(C)

  rz_0 = np.array([0,0,1]).reshape(3, 1)
  rz_base = rz_0 - (np.dot(rz_0.T[0], rx.T[0])) * rx
  rz = rz_base / np.linalg.norm(rz_base)

  ry = np.cross(rz.T[0], rx.T[0])

  R_rect = np.vstack([rx.T, ry.T, rz.T])

  # find the rectification homographices for both images

  H1 = K @ R_rect @ np.linalg.inv(K)
  H2 = K @ R_rect @ R.T @ np.linalg.inv(K)

  return H1, H2

def get_3d_pt_cloud(path):
  img_paths = [img for img in os.listdir(path) if ".jpg" in img]
  #print(img_paths)
  imgs = load_imgs_gray(path, ["view_0.jpg", "view_1.jpg"])
  #path = "/Users/rvolkov/Documents/uni/5561/building-mapper/feature_matching/pouya_test_imgs_2/"
  ##imgs = load_imgs_gray("path", ["Users/rvolkov/Documents/uni/5561/building-mapper/pouya_test_imgs/view_0.png", "Users/rvolkov/Documents/uni/5561/building-mapper/pouya_test_imgs/view_1.png"])
  #imgs = load_imgs_gray(path, ["view_0.png", "view_1.png"])
  #show_imgs(imgs)

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

  E = K2.T @ F @ K1

  # run QR decomp on fundamental matrix

  U, _, V_t = np.linalg.svd(E)

  W_mat = np.array([
     [0, -1, 0],
     [1, 0, 0],
     [0, 0, 1]
  ])

  R1 = U @ W_mat @ V_t
  R2 = U @ W_mat.T @ V_t

  # check determinant of rotation matrices
  if np.linalg.det(R1) < 0:
    R1 = -R1
  if np.linalg.det(R2) < 0:
    R2 = -R2


  t1 = U[:,2].reshape((3, 1))
  t2 = -U[:, 2].reshape((3, 1))

  Rs = [R1, R1, R2, R2]
  Cs = [t1, t2, t1, t2]

  visualize_camera_poses(Rs, Cs)

#  K1 = np.eye(3)
#  K2 = np.eye(3)

  # triangulation
  pts3Ds = []
  P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
  for R, C in zip(Rs, Cs):
    #P2 = K2 @ R @ np.hstack([np.eye(3), -C])
    P2 = K2 @ np.hstack([R, -R @ C])
    pts3D = triangulation(P1, P2, (x1, x2))
    pts3Ds.append(pts3D)

  visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

  # Step 3: disambiguate camera poses
  R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds, K1, (h, w))
  visualize_camera_pose_with_pts(R, C, pts3D)

  print("det(K1):", np.linalg.det(K1))

  # Step 4: rectification
  H1, H2 = compute_rectification(K1, R, C)
  H1 = H1 / H1[2,2]
  H2 = H2 / H2[2,2]
  print("H1:", H1)
  print("H2:", H2)
  img_left_w = my_warp_perspective(im1, H1, (h, w)) # Todo compute warped img left: Hint warp img_left using H1 
  img_right_w = my_warp_perspective(im2, H2, (h2, w2)) # Todo compute warped img right: Hint warp img_left using H1 
  visualize_img_pair(img_left_w, img_right_w)

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


