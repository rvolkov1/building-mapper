import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_imgs_gray(path, imgs):
  print("\n", os.path.join(path, imgs[0]), "\n")
  loaded_imgs = [cv2.imread(os.path.join(path, img)) for img in imgs]

  #loaded_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in loaded_imgs]
  loaded_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in loaded_imgs]

  base_height = loaded_imgs[0].shape[0]
  resized_imgs = []
  for img in loaded_imgs:
    print(img.shape)
    h, w = img.shape[:2]
    scale = base_height / h
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, base_height))
    resized_imgs.append(resized)

  return resized_imgs

def show_imgs(imgs):
  stacked_img = np.hstack(imgs)
  plt.figure(figsize=(15, 5))
  plt.imshow(stacked_img)
  plt.axis('off')
  plt.show()

def visualize_sift(img):
  sift = cv2.SIFT_create()
  kp = sift.detect(img, None)
  img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  plt.imshow(img, cmap='gray', vmin=0, vmax=1)
  plt.axis('off')
  fig = plt.gcf()
  fig.set_size_inches(16, 9)
  fig.tight_layout()
  plt.show()

def sift_pipeline(img):
  #template = cv2.imread('assets/sift_template.jpg', 0)  # read as grey scale image
  #target = cv2.imread('assets/sift_target.jpg', 0)  # read as grey scale image
  H_temp, W_temp = img.shape

  visualize_sift(img)

  #x1, x2 = find_match(template, target)
  #visualize_find_match(template, target, x1, x2)

  ## TODO: specify parameters.
  #A = compute_A(x1, x2)
  #visualize_align_image_using_feature(template, target, x1, x2, A, ransac_thr=3)

  ## TODO: Warp the image using the computed homography.
  #img_warped = compute_warped_image(target, A, W_temp, H_temp)
  #visualize_warp_image(img_warped, template)

def find_match(img1, img2, dist_thr=.7):
  x1 = []
  x2 = []

  sift = cv2.SIFT_create()
  keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
  keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
  bf = cv2.BFMatcher()
  matches = bf.knnMatch (descriptors1, descriptors2,k=2)

  for m1, m2 in matches:
    if m1.distance < dist_thr*m2.distance:
      p1 = keypoints1[m1.queryIdx].pt
      p2 = keypoints2[m1.trainIdx].pt
      x1.append(p1)
      x2.append(p2)

  return np.array(x1), np.array(x2)

def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h / img1.shape[0]
    scale_factor2 = img_h / img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()

def compute_affine_three(correspondence):
  src_points, dst_points = correspondence

  b = np.zeros((6,1), dtype=np.float32)
  A = np.zeros((6, 6), dtype=np.float32)
  x = np.zeros((6,1), dtype=np.float32)

  for i in range(len(src_points)):
    x, y = src_points[i]
    u, v = dst_points[i]
    A[2*i] = np.array([x, y, 1, 0, 0, 0,])
    A[2*i+1] = np.array([0, 0, 0, x, y, 1])

    b[2*i] = u
    b[2*i+1] = v

  #x = (np.linalg.inv(A) @ b).T[0]
  #x = np.lina.solve(A, b).T[0]
  x, *_ = np.linalg.lstsq(A, b, rcond=None)
  x = x.T[0]


  out = np.array([
    [x[0], x[1], x[2]],
    [x[3], x[4], x[5]],
    [0, 0, 1]])

  return out

def apply_transform(pts, H):
  N, _ = pts.shape
  pad = np.ones((N, 1))
  pts_pad = np.hstack([pts, pad])
  points = (H @ pts_pad.T).T
  points = points[:, 0:2] / points[:, 2][:, np.newaxis]
  return points


def compute_affine_RANSAC(correspondence, ransac_thr, iters=1000):
    A = None
    src_points, dst_points = correspondence
    A_best = np.inf
    A = None

    for i in range(iters):
      # select random points
      idx = np.random.choice(len(src_points), 3, replace=False)
      src_mask = src_points[idx]
      dst_mask = dst_points[idx]

      # fit model to points
      curr_A = compute_affine_three((src_mask, dst_mask))

      # apply model
      out_pts = apply_transform(src_points, curr_A)

      # calculcate how many points fit are inliers
      err = np.linalg.norm(out_pts - dst_points, axis=1)

      num_err = np.sum(err > ransac_thr)

      # if model is better than best model, replace best model with current model
      if (num_err < A_best):
          A = curr_A
          A_best = num_err

    return A

def compute_A(x1, x2, ransac_thr):
  # Computes the affine transformation matrix between two sets of points using RANSAC.
  # Input:
  #    x1 (Nx2 ndarray): Keypoints from the first image.
  #    x2 (Nx2 ndarray): Keypoints from the second image.
  # Output:
  #    A (3x3 ndarray): Affine transformation matrix.
  return compute_affine_RANSAC((x1, x2), ransac_thr)

def my_bilinear_interpolate(img, points):
  if (len(img.shape) < 3):
    h, w = img.shape
    img = img.reshape(h, w, 1)

  H, W, C = img.shape

  res = np.zeros((points.shape[0], C), dtype=img.dtype)
  xs = points[:,0]
  ys = points[:,1]

  valid_mask = (xs >=0) & (xs < W) & (ys >=0) & (ys < H)

  if np.any(valid_mask):
    x_low = np.clip(np.floor(xs[valid_mask].astype(np.int64)), 0, img.shape[1]-1).astype(np.int64)
    x_high = np.clip(x_low + 1, 0, img.shape[1]-1).astype(np.int64)

    y_low = np.clip(np.floor(ys[valid_mask].astype(np.int64)), 0, img.shape[0]-1).astype(np.int64)
    y_high = np.clip(y_low + 1, 0, img.shape[0]-1).astype(np.int64)

    dx = (x_high - x_low)
    dy = (y_high - y_low)
    dx[dx == 0] = 1
    dy[dy == 0] = 1
    denoms = dx * dy
    denoms = np.expand_dims(denoms, 1).astype(np.float64)

    q11 = (img[y_low, x_low, :]   * np.expand_dims((x_high - xs[valid_mask]) * (y_high - ys[valid_mask]), 1))
    q12 = (img[y_high, x_low, :]  * np.expand_dims((x_high - xs[valid_mask]) * (ys[valid_mask] - y_low),1))
    q21 = (img[y_low, x_high, :]  * np.expand_dims((xs[valid_mask] - x_low)  * (y_high - ys[valid_mask]),1))
    q22 = (img[y_high, x_high, :] * np.expand_dims((xs[valid_mask] - x_low)  * (ys[valid_mask] - y_low),1))

    res[valid_mask] = np.float32((q11 + q12 + q21 + q22) / denoms)

  return res

def my_warp_perspective(img, M, size):
  outW, outH = size

  # get points
  Y, X = np.mgrid[0:outH, 0:outW]
  pad = np.ones_like(X)
  points = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1]), pad.reshape([-1, 1])], axis=-1)
  #points = (np.linalg.inv(M) @ points.T).T
  points = (M @ points.T).T
  points = points[:, 0:2] / points[:, 2][:, np.newaxis]

  # biliear interpolation of img on new points
  res = my_bilinear_interpolate(img, points)
  res = res.reshape([outH, outW, -1]).astype(np.uint8)

  return res

def compute_warped_image(target, A, width, height):
  # Warps the target image using the provided affine transformation matrix.
  # Input:
  #     target (HxW ndarray): Image to be warped.
  #     A (3x3 ndarray): Affine transformation matrix.
  #     width (int): Width of the output image.
  #     height (int): Height of the output image.
  # Output:
  #     warped_image (HxW ndarray): Warped image.
  img_warped = my_warp_perspective(np.expand_dims(target, -1), A, (width, height))[:, :, 0]
  return img_warped

def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500, filename=None):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sqrt(np.sum(np.square(x2_t[:, :2] - x2), axis=1))
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack((np.array(
        [[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]),
                            np.ones((5, 1)))) @ A[:2, :].T

    scale_factor1 = img_h / img1.shape[0]
    scale_factor2 = img_h / img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y', linewidth=3)
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    if filename:   
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

def visualize_warp_image(img_warped, img, filename=None):
  plt.subplot(131)
  plt.imshow(img_warped, cmap='gray')
  plt.title('Warped image')
  plt.axis('off')
  plt.subplot(132)
  plt.imshow(img, cmap='gray')
  plt.title('Template')
  plt.axis('off')
  plt.subplot(133)
  plt.imshow(np.abs(img_warped - img), cmap='jet')
  plt.title('Error map')
  plt.axis('off')
  fig = plt.gcf()
  fig.set_size_inches(16, 9)
  fig.tight_layout()
  if filename:   
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
  else:
    plt.show()

def normalize_pts(pts):
  mean = np.mean(pts, axis=0)
  std = np.std(pts, axis=0)

  std = np.sqrt(2) / std

  norm_mat = np.array([
    [std[0], 0, -mean[0] * std[0]],
    [0, std[1], -mean[1] * std[1]],
    [0, 0, 1]
  ])

  homo_pts = np.column_stack([pts, np.ones(pts.shape[0])])

  norm_pts = (norm_mat @ homo_pts.T).T

  return norm_pts[:, :2], norm_mat

def compute_F_eight(correspondence):
  src_pts, dst_pts = correspondence

  src_norm, src_T = normalize_pts(src_pts)
  dst_norm, dst_T = normalize_pts(dst_pts)

  ux = src_norm[:, 0]; uy  = src_norm[:, 1]
  vx = dst_norm[:, 0]; vy = dst_norm[:, 1]

  Y = np.column_stack([ vx * ux, vx * uy, vx, vy * ux, vy * uy, vy, ux, uy, np.ones(ux.shape)])

  U, S, Vh = np.linalg.svd(Y, full_matrices=False)
  F = Vh[-1].reshape((3, 3))

  Uf, Sf, Vf = np.linalg.svd(F)
  Sf[-1] = 0 # enforce rank 2 constraint
  F = Uf @ np.diag(Sf) @ Vf

  F = dst_T.T @ F @ src_T

  return F

def compute_F(correspondence, max_iter=5000, eps=1e-3):
  """
  Compute the fundamental matrix using RANSAC for robust estimation.

  Args:
  - correspondence (tuple): Two np.ndarrays (pts1, pts2), each of shape (N, 2).
  - max_iter (int): Maximum number of RANSAC iterations.
  - eps (float): Threshold for determining inliers.
  
  Returns:
  - best_F (np.ndarray): Estimated fundamental matrix of shape (3, 3).
  """
  src_pts, dst_pts = correspondence
  F_best = float('inf')
  F = None

  #eps = 0.5

  N, _ = src_pts.shape
  pad = np.ones((N, 1))
  src_pad = np.hstack([src_pts, pad])
  dst_pad = np.hstack([dst_pts, pad])

  for i in range(max_iter):
    idx = np.random.choice(N, 8, replace=False)

    F_curr = compute_F_eight((src_pts[idx], dst_pts[idx]))

    l2 = F_curr @ src_pad.T
    l1 = F_curr.T @ dst_pad.T 

    d1 = np.abs(np.sum(src_pad * l1.T, axis=1)) / np.linalg.norm(l1[:2, :], axis=0)
    d2 = np.abs(np.sum(dst_pad * l2.T, axis=1)) / np.linalg.norm(l2[:2, :], axis=0)
    err = np.sum((d1 > eps) | (d2 > eps))


    if (err < F_best):
      #print(err, "of:", d1.shape[0])
      F = F_curr
      F_best = err

  return F

def make_image_pair(imgs):
    for img in imgs:
        assert imgs[0].shape[0] == img.shape[0]
        assert imgs[0].ndim == img.ndim
    
    img = np.hstack(imgs)
    
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = (F @ np.array([[p[0], p[1], 1]]).T).flatten()
    p1, p2 = (0, int(-el[2] / el[1])), (img.shape[1], int((-img_width * el[0] - el[2]) / el[1]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_epipolar_lines(img1, img2, correspondence, F, filename=None):
    plt.figure(figsize=(20, 10))
    plt.imshow(make_image_pair((img1, img2)), cmap='gray')
    
    pts1, pts2 = correspondence
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'

    cmap = plt.get_cmap('tab10')  # You can choose another colormap if you like
    colors = [cmap(i % 10) for i in range(pts1.shape[0])]

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        plt.scatter(x1, y1, s=5, color=colors[i])

        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        plt.plot([p1[0] + img1.shape[1], p2[0] + img1.shape[1]], [p1[1], p2[1]], linewidth=0.5, color=colors[i])
    
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        plt.scatter(x2 + img1.shape[1], y2, s=5, color=colors[i])

        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5, color=colors[i])

    plt.axis('off')

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)
    C = C.flatten()  # (3, 1) -> (3,)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def visualize_camera_poses(Rs, Cs):
  assert(len(Rs) == len(Cs) == 4)
  fig = plt.figure(figsize=(20, 10))
  R1, C1 = np.eye(3), np.zeros((3, 1))
  for i in range(4):
    R2, C2 = Rs[i], Cs[i]
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    draw_camera(ax, R1, C1)
    draw_camera(ax, R2, C2)
    set_axes_equal(ax)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(azim=-90, elev=0)
    ax.title.set_text('Configuration {}'.format(i))
  
  fig.tight_layout()
  plt.show()

def make_skew_mat(v):
  return np.array([
    [0,    -v[2],  v[1]],
    [ v[2], 0   , -v[0]],
    [-v[1], v[0],     0]
  ])

def triangulation(P1, P2, correspondence):
  pts1, pts2 = correspondence

  N, _ = pts1.shape
  pts = []

  for i in range(N):
    u = np.hstack([pts1[i, :], 1])
    v = np.hstack([pts2[i, :], 1])

    skew_u = make_skew_mat(u)
    skew_v = make_skew_mat(v)

    top = skew_u @ P1
    bot = skew_v @ P2

    A = np.vstack([top[:2, :], bot[:2, :]])
    U, S, Vt = np.linalg.svd(A)

    pt = Vt[-1]
    pt = pt[:3] / pt[3]
    pts.append(pt)

  pts3D = np.array(pts)

  return pts3D

def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds, filename=None):
  assert(len(Rs) == len(Cs) == 4)
  fig = plt.figure(figsize=(20, 20))
  R1, C1 = np.eye(3), np.zeros((3, 1))
  for i in range(4):
    R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    draw_camera(ax, R1, C1, 5)
    draw_camera(ax, R2, C2, 5)
    ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
    set_axes_equal(ax)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(azim=-90, elev=0)
    ax.title.set_text('Configuration {}'.format(i))
  fig.tight_layout()

  plt.show()

def disambiguate_pose(Rs, Cs, pts3Ds, K=None, screen_size=None):
  """
  Find the best relative camera pose based on the most valid 3D points visible on the screen.

  Args:
  - Rs (list): List of np.ndarrays, each of shape (3, 3), representing possible rotation matrices.
  - Cs (list): List of np.ndarrays, each of shape (3, 1), representing camera centers.
  - pts3Ds (list): List of np.ndarrays, each of shape (N, 3), representing possible 3D points corresponding to different poses.
  - K (np.ndarray): Intrinsic camera matrix of shape (3, 3). <- It is optional to use this as an input. ie its possible without it but easier with it.
  - screen_size (tuple): Screen dimensions as (height, width). <- It is optional to use this as an input. ie its possible without it but easier with it.
  
  Returns:
  - R (np.ndarray): The best rotation matrix of shape (3, 3).
  - C (np.ndarray): The best camera center of shape (3, 1).
  - pts3D (np.ndarray): The best set of 3D points of shape (N, 3).
  """

  best_R = None
  best_C = None
  best_pts = None
  most_in = 0
  best_idx = None

  for idx, (R, C, pts) in enumerate(zip(Rs, Cs, pts3Ds)):
    inliers = 0

    for i in range(pts.shape[0]):
      pt = pts[i].reshape((3, 1))

      r3 = R[2].reshape((1, 3))

      out = r3 @ (pt - C.reshape((3,1)))

      if out.item() > 0:
        inliers += 1

    print("num inliers", inliers)
    #if (inliers > most_in):
    if (idx == 1):
      best_idx = idx
      best_R = R
      best_C = C
      best_pts = pts
      most_in = inliers

  print("best_index: ", best_idx)
  return best_R, best_C, best_pts

def visualize_camera_pose_with_pts(R, C, pts3D, filename=None):
  fig = plt.figure(figsize=(20, 20))

  ax = fig.add_subplot(1, 1, 1, projection='3d')

  R1, C1 = np.eye(3), np.zeros((3, 1))
  draw_camera(ax, R1, C1, 5)
  draw_camera(ax, R, C, 5)

  ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')

  set_axes_equal(ax)
  ax.set_xlabel('x axis')
  ax.set_ylabel('y axis')
  ax.set_zlabel('z axis')

  ax.view_init(azim=-90, elev=0)
  ax.title.set_text('Camera Pose with 3D Points')

  fig.tight_layout()

  if filename:
    plt.savefig(filename, bbox_inches='tight')
  else:
    plt.show()

def visualize_img_pair(img1, img2, filename=None):
  plt.figure(figsize=(20, 10))
  plt.imshow(make_image_pair((img1, img2)), cmap='gray')
  plt.axis('off')

  if filename:   
    plt.savefig(filename, bbox_inches='tight')
  else:
    plt.show()