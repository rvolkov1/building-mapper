import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_imgs_gray(path, imgs):
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


def compute_affine_RANSAC(correspondence, iters=1000):
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

      num_err = np.sum(err > 2.0)

      # if model is better than best model, replace best model with current model
      if (num_err < A_best):
          A = curr_A
          A_best = num_err

    return A

def compute_A(x1, x2):
  # Computes the affine transformation matrix between two sets of points using RANSAC.
  # Input:
  #    x1 (Nx2 ndarray): Keypoints from the first image.
  #    x2 (Nx2 ndarray): Keypoints from the second image.
  # Output:
  #    A (3x3 ndarray): Affine transformation matrix.
  return compute_affine_RANSAC((x1, x2))

def my_bilinear_interpolate(img, points):
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