import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision.transforms.functional

from mast3r.mast3r.model import AsymmetricMASt3R
from mast3r.mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.dust3r.dust3r.inference import inference
from mast3r.dust3r.dust3r.utils.image import load_images

from feature_matching.feature_matching_utils import load_imgs_gray, show_imgs, visualize_sift, find_match, visualize_find_match, compute_F, visualize_epipolar_lines, compute_camera_pose, visualize_camera_poses, triangulation, visualize_camera_poses_with_pts, disambiguate_pose, visualize_camera_pose_with_pts, my_warp_perspective, visualize_img_pair, dense_match, visualize_disparity_map

def get_dl_correspondences(im1, im2):
  device = 'cpu'
  schedule = 'cosine'
  lr = 0.01
  niter = 300

  model_name = "mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
  # you can put the path to a local checkpoint in model_name if needed
  model = AsymmetricMASt3R.from_pretrained(model_name).to(device)


  #path = "/Users/rvolkov/Documents/uni/5561/building-mapper/zind_subset/0528/2d_views/room 01/corres_0"
  images = load_images([im1, im2], size=512)
  output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

  # at this stage, you have the raw dust3r predictions
  view1, pred1 = output['view1'], output['pred1']
  view2, pred2 = output['view2'], output['pred2']

  desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

  # find 2D-2D matches between the two images
  matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                  device=device, dist='dot', block_size=2**13)

  # ignore small border around the edge
  H0, W0 = view1['true_shape'][0]
  valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
      matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

  H1, W1 = view2['true_shape'][0]
  valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
      matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

  valid_matches = valid_matches_im0 & valid_matches_im1
  matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

  n_viz = 20
  num_matches = matches_im0.shape[0]
  match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
  viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

  image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
  image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

  viz_imgs = []
  for i, view in enumerate([view1, view2]):
      rgb_tensor = view['img'] * image_std + image_mean
      viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

  H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
  img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
  img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
  img = np.concatenate((img0, img1), axis=1)

  #plt.figure()
  #plt.imshow(img)
  #cmap = plt.get_cmap('jet')
  #for i in range(n_viz):
  #    (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
  #    plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)

  #pl.show(block=True)

  return (viz_matches_im0, viz_matches_im1)

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
  theta_x = np.deg2rad(86.0)
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

def get_dl_recon(path):
  img_paths = [img for img in os.listdir(path) if ".jpg" in img]
  print(img_paths)
  #imgs = load_imgs_gray(path, ["view_0.jpg", "view_1.jpg"])
  #path = "/Users/rvolkov/Documents/uni/5561/building-mapper/feature_matching/pouya_test_imgs_2/"
  #imgs = load_imgs_gray("path", ["Users/rvolkov/Documents/uni/5561/building-mapper/pouya_test_imgs/view_0.png", "Users/rvolkov/Documents/uni/5561/building-mapper/pouya_test_imgs/view_1.png"])

  im_path_1 = path + "/view_0.jpg"  
  im_path_2 = path + "/view_1.jpg" 

  print("path:", im_path_1)
  print("path:", im_path_2)
  print(os.path.exists(im_path_1))
  print(os.path.exists(im_path_2))

  imgs = load_imgs_gray(path, ["view_0.jpg", "view_1.jpg"])
  show_imgs(imgs)

  im1 = imgs[0]
  im2 = imgs[1]

  #x1, x2, descriptors1, descriptors2 = find_match(im1, im2, dist_thr=0.95)

  x1, x2 = get_dl_correspondences(im_path_1, im_path_2)

  visualize_find_match(im1, im2, x1, x2)

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
    P2 = K1 @ np.hstack([R, -R @ C])
    pts3D = triangulation(P1, P2, (x1, x2))
    pts3Ds.append(pts3D)

  visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

  # Step 3: disambiguate camera poses
  R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds, K1, (h, w))
  visualize_camera_pose_with_pts(R, C, pts3D)

  #return R, C, pts3D

  # Step 4: rectification
  H1, H2 = compute_rectification(K1, R, C)
  #H1 = H1 / H1[2,2]
  #H2 = H2 / H2[2,2]
  print("H1:", H1)
  print("H2:", H2)
  img_left_w = my_warp_perspective(im1, H1, (h, w)) # Todo compute warped img left: Hint warp img_left using H1 
  img_right_w = my_warp_perspective(im2, H2, (h2, w2)) # Todo compute warped img right: Hint warp img_left using H1 
  visualize_img_pair(img_left_w, img_right_w)

  disparity = dense_match(img_left_w, img_right_w, descriptors1, descriptors1)
  visualize_disparity_map(disparity)
