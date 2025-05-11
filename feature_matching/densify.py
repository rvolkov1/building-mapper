import open3d as o3d
import numpy as np


def pixel_to_ray(u, v, K):
    """Return a 3‑D ray direction (unit) in camera coords from pixel (u,v)."""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray = np.array([x, y, 1.0])
    return ray / np.linalg.norm(ray)

def intersect_plane(ray_cam, R, t, n, d):
    """
    ray_cam : unit vector in camera space
    R,t     : camera→world (3×3, 3×1)
    n,d     : plane in world coords (n·X + d = 0)
    """
    # Ray origin in world coords (camera centre)
    C = -R.T @ t
    C = C.reshape(3)
    # Ray direction in world coords
    dir_world = R.T @ ray_cam
    denom = n @ dir_world
    if abs(denom) < 1e-6:
        return None     # parallel
    s = -(n @ C + d) / denom
    if s <= 0:
        return None     # behind camera
    return C + s * dir_world

def densify_from_mask(mask, K, R, t, planes, step=2):
    """
    Iterate over every 'step' pixels to keep runtime bounded.
    Returns (M,3) xyz and (M,3) RGB (dummy white here).
    """
    H, W = mask.shape
    xyz = []
    for v in range(0, H, step):
        for u in range(0, W, step):
            ℓ = mask[v, u]
            if ℓ not in planes:
                continue
            n, d = planes[ℓ]
            ray = pixel_to_ray(u, v, K)
            Xw = intersect_plane(ray, R, t, n, d)
            if Xw is not None:
                xyz.append(Xw)
    xyz = np.array(xyz)
    colors = np.ones_like(xyz)    # all white – replace if you want
    return xyz, colors

from open3d.visualization import rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D             # noqa

def save_png_via_matplotlib(X, colors, out_png="densified_matplotlib.png",
                            elev=20, azim=-45, s=0.5):
    """
    X       : (N,3) ndarray
    colors  : (N,3) float in 0‑1
    """
    fig = plt.figure(figsize=(8,6), dpi=150)
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=colors, s=s)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"[Matplotlib] snapshot written → {out_png}")