import numpy as np
import matplotlib.pyplot as plt

def perspective_to_dir(x, y, w, h, fov_deg, yaw_deg, pitch_deg):
    '''
    take a point in perspective space and convert it to a direction ray
    '''
    fov = np.deg2rad(fov_deg)
    fx = fy = 0.5 * w / np.tan(0.5 * fov)
    cx = w / 2
    cy = h / 2

    # camera-space direction
    dx = (x - cx) / fx
    dy = -(y - cy) / fy
    dz = 1.0
    d = np.array([dx, dy, dz])
    d /= np.linalg.norm(d)

    #rotation matrices
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    R = R_yaw @ R_pitch
    return R @ d #final ray

def dir_to_eq_coords(dir, im_w, im_h):
    '''
    convert a direction ray to equirectangular coordinates
    '''
    x, y, z = dir
    lon = np.arctan2(x, z)
    lat = np.arcsin(y)

    u = (lon / np.pi + 1) / 2 * im_w
    v = (0.5 - lat / np.pi) * im_h
    return (u, v)

def perspective_to_spherical(x, y, w, h, fov, yaw, pitch, eq_w, eq_h):
    '''
    convert perspective point to a spherical point
    '''
    dir = perspective_to_dir(x, y, w, h, fov, yaw, pitch)
    u, v = dir_to_eq_coords(dir, eq_w, eq_h)
    return int(u), int(v)

def visualize_correspondence_pano(im, correspondence):
    '''
    adjusted hw2 utility
    '''
    src_points, dst_points = correspondence
    #src_image, dst_image = pano["images"][src_index], pano["images"][dst_index]

    #combined_image = np.hstack((src_image, dst_image))
    #dst_points_adjusted = dst_points + np.array([src_image.shape[1], 0])

    fig, ax = plt.subplots()
    ax.imshow(im)

    ax.scatter(src_points[:, 0], src_points[:, 1], color='blue', label='Source Points', s=100, edgecolors='k')
    ax.scatter(dst_points[:, 0], dst_points[:, 1], color='red', label='Destination Points', s=100, edgecolors='k')

    for src, dst in zip(src_points, dst_points):
        ax.plot([src[0], dst[0]], [src[1], dst[1]], 'k--', lw=1)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title('Ceiling to Floor Point Correspondences')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def transform_util(H, points):
    '''
    credit goes to hw2 for this function
    '''
    points_h = np.hstack((points, np.ones((len(points), 1))))
    dist_points_proj = (H @ points_h.T).T
    dist_points_proj = dist_points_proj / dist_points_proj[:, 2][:, None]
    return dist_points_proj[:, :2]

def spherical_to_perspective(u, v, W, H, fov_deg, yaw_deg=0, pitch_deg=0, w=512, h=512):
    """ 
    function obtained from internet resources.

    map points (u, v) in a spherical image to (x, y) in a perspective view.

    Parameters:
        u, v      point in equirectangular image (pixel)
        W, H      dimensions of the equirectangular image
        fov_deg   horizontal FOV of the perspective image
        yaw_deg   yaw of the perspective view (azimuth)
        pitch_deg pitch of the perspective view (elevation)
        w, h      dimensions of the perspective image
 
    Returns:
        (x, y) in perspective view, or None if point lies outside FOV
    """
    #equirectangular to direction vector
    lon = (u / W) * 2 * np.pi - np.pi        
    lat = np.pi / 2 - (v / H) * np.pi         

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    dir_world = np.stack([x, y, z], axis=1)

    # Rotate into perspective camera space
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    R_yaw = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [ 0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])
    R = R_pitch @ R_yaw
    dir_cam = dir_world @ R.T   # transform into camera-local frame

    # Step 3: perspective projection
    fov = np.deg2rad(fov_deg)
    fx = fy = 0.5 * w / np.tan(0.5 * fov)

    z_cam = dir_cam[:, 2]
    x_img = fx * (dir_cam[:, 0] / z_cam) + w / 2
    y_img = fy * (-dir_cam[:, 1] / z_cam) + h / 2

    # Step 4: mask invalid (behind or outside image bounds)
    visible = (z_cam > 0) & (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)

    x_img[~visible] = np.nan
    y_img[~visible] = np.nan

    return (x_img, y_img)
