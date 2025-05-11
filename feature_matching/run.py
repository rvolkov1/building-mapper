import os 
import re
import sys
sys.path.append("/building-mapper/mast3r")
import matplotlib.pyplot as plt


from recon_3d import get_dl_recon
from densify import densify_from_mask, save_png_via_matplotlib

import open3d as o3d
import numpy as np


if __name__ == "__main__":
  path = '/building-mapper/pairs/pair3'
  K1, R1, t1, K2, R2, t2, X, cols, lbl_sparse, mask1, mask2 = get_dl_recon(path)


  planes = {}     # label ℓ → (normal n, distance d)

  for l in [1,2,3,4,5]:
      pts = X[lbl_sparse == l]
      if len(pts) < 3:
          continue
      pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
      # RANSAC plane: returns (a,b,c,d), inliers
      model, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
      n = np.array(model[:3])
      d = model[3]
      planes[l] = (n, d)


  xyz1, col1 = densify_from_mask(mask1, K1, R1, t1, planes)
  xyz2, col2 = densify_from_mask(mask2, K2, R2, t2, planes)

  print("X.shape:", X.shape)
  print("xyz1.shape:", xyz1.shape)
  
  xyz_all   = np.vstack([X, xyz1, xyz2])
  colors_all = np.vstack([cols, col1, col2])

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz_all)
  pcd.colors = o3d.utility.Vector3dVector(colors_all)

  # Optional clean‑ups
  # pcd = pcd.voxel_down_sample(voxel_size=0.01)
  # pcd.estimate_normals()

  save_png_via_matplotlib(xyz_all[:10_000], colors_all[:10_000])

  o3d.io.write_point_cloud("densified.ply", pcd)
  print("Saved densified.ply  ({} points)".format(np.asarray(pcd.points).shape[0]))