import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision.transforms.functional

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
import dill as pickle

# from feature_matching.feature_matching_utils import load_imgs_gray, show_imgs, visualize_sift, find_match, visualize_find_match, compute_F, visualize_epipolar_lines, compute_camera_pose, visualize_camera_poses, triangulation, visualize_camera_poses_with_pts, disambiguate_pose, visualize_camera_pose_with_pts, my_warp_perspective, visualize_img_pair, dense_match, visualize_disparity_map


device = 'cuda'
schedule = 'cosine'
lr = 0.01
niter = 300

def get_mast3r_preds(im1, im2):
  model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
  model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

  images = load_images([im1, im2], size=512)
  output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

  # at this stage, you have the raw dust3r predictions
  view1, pred1 = output['view1'], output['pred1']
  view2, pred2 = output['view2'], output['pred2']

  scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
  loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

  return scene, pred1, pred2, view1, view2


CACHE_F = "mast3r_preds.pkl" # pick a location you like

def load_or_run(im_path_1, im_path_2, cache_f=CACHE_F):
    if os.path.exists(cache_f):
        with open(cache_f, "rb") as f:
            return pickle.load(f)

    scene, pred1, pred2, view1, view2 = get_mast3r_preds(im_path_1, im_path_2)
    with open(cache_f, "wb") as f:
        pickle.dump((scene, pred1, pred2, view1, view2), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return scene, pred1, pred2, view1, view2

def get_intrinsics(scene):
  intrinsics = scene.get_intrinsics()
  K1, K2 = intrinsics[0].cpu().detach().numpy(), intrinsics[1].cpu().detach().numpy()
  return K1, K2

def get_mast3r_correspondences(pred1, pred2, view1, view2):
  
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

  num_matches = matches_im0.shape[0]
  n_viz = num_matches
  print("num_matches: ", num_matches)
  match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
  viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

  image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
  image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

  viz_imgs = []
  for i, view in enumerate([view1, view2]):
      rgb_tensor = view['img'] * image_std + image_mean
      viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

  H0, W0, H1, W1 = *viz_imgs[0].shape[:2], * viz_imgs[1].shape[:2]
  img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
  img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
  img = np.concatenate((img0, img1), axis=1)

  plt.figure()
  # plt.imshow(img)
  cmap = plt.get_cmap('jet')
  corres_1, corres_2 = [], [] 
  for i in range(n_viz):
     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
     corres_1.append((x0, y0))
     corres_2.append((x1, y1))
  
  corres_1 = np.array(corres_1)
  corres_2 = np.array(corres_2)

  print(corres_1.shape, corres_2.shape)

    #  plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)

  # plt.savefig("output_matches.png", dpi=300, bbox_inches='tight')
  # plt.show(block=True)

  return (corres_1, corres_2)

def get_dl_recon(path):
  img_paths = [img for img in os.listdir(path) if ".png" in img]
  print(img_paths)
  #imgs = load_imgs_gray(path, ["view_0.jpg", "view_1.jpg"])
  #path = "/Users/rvolkov/Documents/uni/5561/building-mapper/feature_matching/pouya_test_imgs_2/"
  #imgs = load_imgs_gray("path", ["Users/rvolkov/Documents/uni/5561/building-mapper/pouya_test_imgs/view_0.png", "Users/rvolkov/Documents/uni/5561/building-mapper/pouya_test_imgs/view_1.png"])

  im_path_1 = path + "/view_0.png"  
  im_path_2 = path + "/view_1.png"

  # ------------------------------------------------------------------
  # 0.  Load inputs ---------------------------------------------------
  # ------------------------------------------------------------------
  img1 = cv2.imread(im_path_1)        # BGR order!
  img2 = cv2.imread(im_path_2)

  # print("path:", im_path_1)
  # print("path:", im_path_2)
  # print(os.path.exists(im_path_1))
  # print(os.path.exists(im_path_2))

  # imgs = load_imgs_gray(path, ["view_0.png", "view_1.png"])
  # show_imgs(imgs)

  # im1 = imgs[0]
  # im2 = imgs[1]

  #x1, x2, descriptors1, descriptors2 = find_match(im1, im2, dist_thr=0.95)
  scene, pred1, pred2, view1, view2 = load_or_run(im_path_1, im_path_2)
  K1, K2 = get_intrinsics(scene)
  pts1, pts2 = get_mast3r_correspondences(pred1, pred2, view1, view2)

  # visualize_find_match(im1, im2, x1, x2)

  # ------------------------------------------------------------------
  # 1.  Robust F (pixel domain) --------------------------------------
  # ------------------------------------------------------------------
  F, inliers = cv2.findFundamentalMat(
      pts1, pts2,
      method=cv2.FM_RANSAC,
      ransacReprojThreshold=1.0,
      confidence=0.999
  )
  inliers = inliers.ravel().astype(bool)
  pts1_i, pts2_i = pts1[inliers], pts2[inliers]

  # ------------------------------------------------------------------
  # 2.  Essential matrix (uses intrinsics) ---------------------------
  # ------------------------------------------------------------------
  E = K2.T @ F @ K1

  # ------------------------------------------------------------------
  # 3.  Recover relative pose R,t  -----------------------------------
  #     Input points *must* be normalised by intrinsics here
  # ------------------------------------------------------------------
  _, R, t, mask_pose = cv2.recoverPose(E, pts1_i, pts2_i, K1)

  pts1_i = np.ascontiguousarray(pts1_i, dtype=np.float32)
  pts2_i = np.ascontiguousarray(pts2_i, dtype=np.float32)

    # Optionally refine inliers:
  # pts1_i, pts2_i = pts1_i[mask_pose.ravel() == 1], pts2_i[mask_pose.ravel() == 1]

  # ------------------------------------------------------------------
  # 4.  Triangulate ---------------------------------------------------
  #     Build projection matrices in *normalized* camera coords
  # ------------------------------------------------------------------
  P1 = np.hstack((np.eye(3),  np.zeros((3, 1))))
  P2 = np.hstack((R,          t))          # t is unit‑length (scale unknown)


  # Convert pixel → normalized image coordinates
  pts1_n = cv2.undistortPoints(pts1_i.reshape(-1, 1, 2), K1, None).reshape(-1, 2).T
  pts2_n = cv2.undistortPoints(pts2_i.reshape(-1, 1, 2), K2, None).reshape(-1, 2).T

  # Triangulate (homogeneous → Euclidean)
  X_h = cv2.triangulatePoints(P1, P2, pts1_n, pts2_n)        # (4,N)
  X = (X_h / X_h[3])[:3].T                                   # (N,3)

  # Keep points with positive depth for both cameras
  depth1 =  X @  P1[2, :3] + P1[2, 3]
  depth2 = (X @  R.T @ np.array([0, 0, 1])) + t[2]
  mask_z = (depth1 > 0) & (depth2 > 0)
  X = X[mask_z]
  pts1_vis = pts1_i[mask_z]          # to sample colour

  # ------------------------------------------------------------------
  # 5.  Compute reprojection error (optional filtering) --------------
  # ------------------------------------------------------------------
  proj1 = (K1 @ P1 @ np.vstack((X.T, np.ones((1, X.shape[0]))))).T
  proj1 = (proj1[:, :2].T / proj1[:, 2]).T
  err   = np.linalg.norm(proj1 - pts1_vis, axis=1)
  good  = err < 2.0          # px threshold
  X     = X[good]
  pts1_vis = pts1_vis[good]

  # ------------------------------------------------------------------
  # 6.  Grab colour from img1  ---------------------------------------
  # ------------------------------------------------------------------
  pts1_vis[:, 1].astype(int)
  pts1_vis[:, 0].astype(int)
  print("type(img1): ", type(img1))
  cols = img1[pts1_vis[:, 1].astype(int), pts1_vis[:, 0].astype(int), ::-1]  # BGR→RGB
  print("cols.shape:", cols.shape)
  cols = cols / 255.0

  # ------------------------------------------------------------------
  # 7.  Visualise -----------------------------------------------------
  # ------------------------------------------------------------------
  fig = plt.figure(figsize=(8,6))
  ax  = fig.add_subplot(111, projection='3d')
  ax.scatter(X[:,0], X[:,1], X[:,2], s=3, c=cols, depthshade=False)
  print("X.shape:", X.shape)
  print("cols.shape: ", cols.shape)
  ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
  ax.set_title("Sparse 3‑D reconstruction (scale up to unknown factor)")
  plt.tight_layout()
  # plt.show()
  plt.savefig("sparse_3d_recon.png")
  
  save_ply_ascii("sparse_recon_ascii.ply", X, cols)


  mask1, mask2 = np.load("/building-mapper/segmentation/view_0.npy"), np.load("/building-mapper/segmentation/view_1.npy")
  print("mask1.shape: ", mask1.shape)
  lbl_sparse = np.zeros((pts1_vis.shape[0]))
  pts1_vis_ = pts1_vis.copy().astype('uint8')
  for i in range(pts1_vis.shape[0]): 
    lbl_sparse[i] = mask1[pts1_vis_[i][0], pts1_vis_[i][1]]
  print("lbl_sparse.shape:", lbl_sparse.shape)
  print("X.shape:", X.shape)

  return K1, np.eye(3), np.zeros_like(t), K2, R, t, X, cols, lbl_sparse, mask1, mask2


def save_ply_ascii(path, xyz, rgb):
    n = xyz.shape[0]
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(xyz, (rgb * 255).astype(np.uint8)):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
