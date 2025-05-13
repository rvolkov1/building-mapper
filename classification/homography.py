'''
Author: Dylan Blake
'''
import numpy as np
from utils import visualize_correspondence_pano, transform_util
def homography_hypothesis(correspondence):
    '''
    N point homography algo to find the mapping from src to dst
    '''
    src_points, dst_points = correspondence
    N = src_points.shape[0]
    #concat a 1 to the end of each point
    s = np.concatenate((src_points, np.ones((src_points.shape[0], 1))), axis=1)
    
    #form matrix A
    A = np.ndarray(shape=(2*N, 9))
    for i in range(N):
        u, v = dst_points[i]
        A[2*i] = np.hstack((s[i, :], np.zeros(3),  -u * s[i, :]))
        A[2*i+1] = np.hstack((np.zeros(3), s[i, :],  -v * s[i, :]))

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = np.reshape(h, (3,3))
    return H

def stochastic_correspondences(p_labels_fwc, horizon_y, iters=1024):
    '''
    randomly sample from the ceiling points.
    we assume vertical scenes, so we can map ceiling onto floor given the current labeling by reflecting a ceiling point across the horizon point to a floor point
    '''
    from classify import Label
    im_h, im_w = p_labels_fwc.shape[:2]
    matching_region_extent = im_h - horizon_y
    ceiling_low = horizon_y - matching_region_extent #anything above this in the image has no hope of a correspondence on the floor within the image
    
    src_pts = []
    dst_pts = []
    for i in range(iters):
        cm = np.random.randint(ceiling_low, horizon_y, size=1)[0]
        cn = np.random.randint(0, im_w, size=1)[0]
        if p_labels_fwc[cm, cn] == Label.CEILING:
            #reflect across horizon line
            fm = (horizon_y + (horizon_y - cm))
            fn = cn
            if fm < im_h and p_labels_fwc[fm, fn] == Label.FLOOR:
                src_pts.append(np.array([cn,cm], dtype=np.float64))
                dst_pts.append(np.array([fn,fm], dtype=np.float64))
    return (np.array(src_pts), np.array(dst_pts))

def homography_RANSAC(perspective_im, p_labels_fwc, horizon_y, iters=128):
    '''
    we require perspective space data. anything in spherical space must be transformed prior to calling this function

    Input: 
        p_labels_fwc (perspective space): pixel-wise floor-wall-ceiling labels
        max_ceiling (perspective space): max ceiling y coordinate - all pixels labeled ceiling are below this
        min_floor (perspective space): min floor y coordinate - all pixels labeled floor are above this
        horizon_y (perspective space): we assume vertical images (no camera tilt), thus the horizon line is horizontal, 
                   therefore we only need a y coordinate to represent the horizon line, across which
                   we reflect a randomly selected ceiling point to a floor point 
    Output:
        H: best scoring homography mapping ceiling to floor
    '''
    im_h = p_labels_fwc.shape[0]
    assert horizon_y < im_h, "horizon must be below image height"

    epsilon = 0.001
    src_points, dst_points = stochastic_correspondences(p_labels_fwc, horizon_y)

    visualize_correspondence_pano(perspective_im, (src_points, dst_points))

    max_inliers = 0
    best_H = None
    for i in range(iters):
        sampler = np.random.choice(np.arange(src_points.shape[0]), 4, replace=False)
        src_sample = src_points[sampler]
        dst_sample = dst_points[sampler]
        candidate_H = homography_hypothesis((src_sample, dst_sample))

        #calculate inliers
        diff = np.linalg.norm(dst_points - transform_util(candidate_H, src_points), axis=1)
        inlier_mask = diff < epsilon
        inlier_count = np.sum(inlier_mask)
        if inlier_count > max_inliers:
            best_H = candidate_H
            max_inliers = inlier_count
    return best_H
