import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import matplotlib as mpl
from scipy.spatial import ConvexHull
import cv2
from itertools import combinations


def GenerateFloorplan(planNum, N=10, sigma=1, rngSeed=None):
    if planNum == 1:
        corner1 = [0, 0]
        corner2 = [10, 0]
        corner3 = [10, 10]
        corner4 = [0, 10]
        endpoints = [corner1, corner2, corner3, corner4, corner1]
    if planNum == 2:
        corner1 = [0, 0]
        corner2 = [10, 0]
        corner3 = [10, 10]
        corner4 = [5, 15]
        corner5 = [0, 10]
        endpoints = [corner1, corner2, corner3, corner4, corner5, corner1]
    if planNum == 3:
        corner1 = [0, 0]
        corner2 = [10, 0]
        corner3 = [10, 10]
        corner4 = [-3, 10]
        corner5 = [-3, 5]
        corner6 = [0, 5]
        endpoints = [corner1, corner2, corner3, corner4, corner5, corner6, corner1]
        
    numPairs = len(endpoints) - 1

    pts = np.zeros((N*numPairs, 2))
    # Generate points between each pair of endpoints
    for i in range(numPairs):
        pts[i*N:i*N + N, 0] = np.linspace(endpoints[i][0], endpoints[i+1][0], N)
        pts[i*N:i*N + N, 1] = np.linspace(endpoints[i][1], endpoints[i+1][1], N)

    rng = np.random.default_rng(rngSeed)
    noise = rng.normal(scale=sigma, size=(N*numPairs, 2))

    return pts, pts + noise


def Cross2(a, b):
    a = a.flatten()
    b = b.reshape(2, 1)
    a_x = np.array([-a[1], a[0]]).reshape(1, 2)
    return a_x @ b


def CalculateDistFromLine(l, x):
    # l (list)
    # x (ndarray): (2, N)
    
    a = l[0]
    b = l[1]
    
    b_x = np.array([[-b[1,0], b[0,0]]])
    d = b_x @ (x - a)

    return np.abs(d).flatten()


def GetLine(x1, x2):
    """ Calculates a line from two points 
    """
    a = x1
    b = x2 - x1
    b = b/np.linalg.norm(b)
    return [a.reshape(2, 1), b.reshape(2, 1)]


def GetLinePts(l, x_lim, y_lim):
    # Choose the coefficient that captures the most of x_lim or y_lim
    lbda_min = np.min(( (x_lim[0] - l[0][0])/l[1][0], (y_lim[0] - l[0][1])/l[1][1] ))
    lbda_max = np.max(( (x_lim[1] - l[0][0])/l[1][0], (y_lim[1] - l[0][1])/l[1][1] ))
    if abs(l[1][1]) < 1e-9:
        lbda_min = (x_lim[0] - l[0][0])/l[1][0]
        lbda_max = (x_lim[1] - l[0][0])/l[1][0]
    elif abs(l[1][0]) < 1e-9:
        lbda_min = (y_lim[0] - l[0][1])/l[1][1]
        lbda_max = (y_lim[1] - l[0][1])/l[1][1]

    lbda = np.array([lbda_min, lbda_max]).reshape(1, 2)
    r = l[0] + l[1] @ lbda
    return r[0], r[1]


def FitLine_ransac(pts, thr_d=0.8, maxIter=10000, alpha=1, beta=1):
    N = pts.shape[0]
    idxList = np.arange(N)
    l_best = []
    num_inliers_best = 0
    val_to_max = 0
    inliers_best = np.zeros(N)

    rng = np.random.default_rng()

    iter = 0
    while num_inliers_best < N and iter < maxIter:
        # Select two points
        idxChoice = rng.choice(idxList, 2, replace=False)
        pt1 = pts[idxChoice[0]]
        pt2 = pts[idxChoice[1]]

        l = GetLine(pt1, pt2)
        d = CalculateDistFromLine(l, pts.T)
        num_inliers = np.sum(d <= thr_d)
        inlier_points = pts[d <= thr_d, :]

        x_lim = np.array([np.min(inlier_points[:, 0]), np.max(inlier_points[:, 0])])
        y_lim = np.array([np.min(inlier_points[:, 1]), np.max(inlier_points[:, 1])])
        x, y = GetLinePts(l, x_lim, y_lim)
        l.append([x, y])
        # fig, ax = plt.subplots()
        # ax.scatter(pts[:, 0], pts[:, 1], c="r")
        # ax.scatter(pts[np.abs(d) <= thr_d, 0], pts[np.abs(d) <= thr_d, 1], c="g")
        # ax.plot(x, y, c="g", ls="-")
        # pad = 1
        # ax.set_xlim((np.min(pts[:, 0]) - pad, np.max(pts[:, 0]) + pad))
        # ax.set_ylim((np.min(pts[:, 1]) - pad, np.max(pts[:, 1]) + pad))
        # ax.set_aspect('equal', adjustable='box')
        # # ax[1].set_aspect('equal')
        #
        # # ax[2].hist(ptsRotated[:, 0], bins=nBins)[0]
        # plt.show()


        U = CalculateUniformity(l, pts, thr_d=thr_d)
        obj_fun = alpha*num_inliers/N + beta*U
        if obj_fun > val_to_max:
            num_inliers_best = num_inliers
            val_to_max = obj_fun
            l_best = l
            inliers_best = np.abs(d) <= thr_d

        iter += 1

    return l_best, val_to_max, inliers_best
        

def CalculateUniformity(l, pts, thr_d=0.8, showOutput=False):
    # Check for how the inliers are distributed along the line l
    # d = CalculateDistFromLine(l, pts.reshape(2, -1))
    d = CalculateDistFromLine(l, pts.T)

    inlier_selection = d <= thr_d
    ptsInliers = pts[inlier_selection, :]
    lineAngle = np.atan2(l[1][1], l[1][0])[0]
    rotationAngle = lineAngle
    ct = np.cos(rotationAngle)
    st = np.sin(rotationAngle)
    DCM = np.array([[ct, st], [-st, ct]])
    ptsRotated = (DCM @ ptsInliers.T).T
    l_rotated = [DCM @ l[0], DCM @ l[1]]

    # Shift to center of mass
    numInliers = ptsInliers.shape[0]
    if numInliers == 0:
        return -1
    x_c = np.sum(ptsRotated[:, 0])/numInliers
    y_c = np.sum(ptsRotated[:, 1])/numInliers
    ptsRotated[:, 0] -= x_c
    ptsRotated[:, 1] -= y_c
    l_rotated[0][0] -= x_c
    l_rotated[0][1] -= y_c

    # Calculate the moment of inertia
    Ixx = np.sum(ptsRotated[:, 1]**2)/numInliers
    Iyy = np.sum(ptsRotated[:, 0]**2)/numInliers
    Ixy = np.sum(ptsRotated[:, 0]*ptsRotated[:, 1])/numInliers

    # Uniformity score (kind of like pearson test statistic)
    nBins = 10
    h = np.histogram(ptsRotated[:, 0], bins=nBins)[0]
    E = numInliers/nBins
    # The 0.9 is totally empirical
    U = 1 - np.sum((h - E)**2)/(numInliers - E)**2 - 0.9
    # U = 1 - np.sum((h - E)**2)/E**2

    if showOutput == True:
        momentText = f"Ixx = {Ixx}, Iyy = {Iyy}, Ixy = {Ixy}"
        print(momentText)
        print(f"U = {U}")

        x_lim = np.array([np.min(ptsInliers[:, 0]), np.max(ptsInliers[:, 0])])
        y_lim = np.array([np.min(ptsInliers[:, 1]), np.max(ptsInliers[:, 1])])
        x_lim_rotated = np.array([np.min(ptsRotated[:, 0]), np.max(ptsRotated[:, 0])])
        y_lim_rotated = np.array([np.min(ptsRotated[:, 1]), np.max(ptsRotated[:, 1])])

        x, y = GetLinePts(l, x_lim, y_lim)
        x_rotated, y_rotated = GetLinePts(l_rotated, x_lim_rotated, y_lim_rotated)
        
        fig, ax = plt.subplots(1, 2)
        ax[0].scatter(pts[:, 0], pts[:, 1], c="r")
        ax[0].scatter(ptsInliers[:, 0], ptsInliers[:, 1], c="g")
        ax[0].plot(x, y, c="g", ls="-")
        ax[1].scatter(ptsRotated[:, 0], ptsRotated[:, 1], c="g")
        ax[1].plot(x_rotated, y_rotated, c='g', ls='-')
        pad = 1
        ax[0].set_xlim((np.min(pts[:, 0]) - pad, np.max(pts[:, 0]) + pad))
        ax[0].set_ylim((np.min(pts[:, 1]) - pad, np.max(pts[:, 1]) + pad))
        ax[0].set_aspect('equal', adjustable='box')
        # ax[1].set_aspect('equal')

        # ax[2].hist(ptsRotated[:, 0], bins=nBins)[0]
        plt.show()

    return U


def FitMultipleLines(pts, thr_d=0.8, alpha=1, beta=1, minOutlierFrac=0.1):
    N = pts.shape[0]
    ptsAvail = np.copy(pts)
    outlierFrac = 1
    l = []
    inliers = []
    inlierPts = []
    obj_fun = []

    while outlierFrac > minOutlierFrac:
        l_i, obj_fun_i, inliers_i = FitLine_ransac(ptsAvail, thr_d=thr_d, alpha=alpha, beta=beta)
        inlierPts.append(ptsAvail[inliers_i])
        inliers.append(inliers_i)
        ptsAvail = np.copy(ptsAvail[np.logical_not(inliers_i)])
        l.append(l_i)
        obj_fun.append(obj_fun_i)
        # Calculate fraction of inliers
        numInliers = 0
        for k in range(len(inliers)):
            numInliers += np.sum(inliers[k])
        outlierFrac = (N - numInliers)/N

        print(outlierFrac)
    
    return l, obj_fun, inlierPts


def FindVertices(lineList, inliers, inlierRadius=0.5, radius_expansion_factor=2, max_num_expansions=5, max_ep_dist=0.5):
    # Each line should have two end points, therefore we look for two "real" intersections between each line element.
    # There would have to be some consistency check, as lines that don't correspond to corners may intersect at further away points.

    # Compute intersections
    numLines = len(lineList)
    liXlj = np.zeros((numLines, numLines, 2))
    points_found = []
    for i, li in enumerate(lineList):
        for j, lj in enumerate(lineList):
            a1 = li[0]
            b1 = li[1]
            a2 = lj[0]
            b2 = lj[1]
            num = Cross2(b1, a1 - a2)
            den = Cross2(b1, b2)
            if np.abs(den) < 1e-9:
                liXlj[i, j] = np.array([np.inf, np.inf])
                points_found.append([np.inf, np.inf])
            else:
                lbda = num/den
                liXlj[i, j] = (a2 + lbda*b2).flatten()
                points_found.append((a2 + lbda*b2).flatten().tolist())

    unique_pts = np.unique(np.round(points_found, decimals=5), axis=0)

    # Stores the index of the corresponding point in unique_pts for each line
    vertex_candidates = np.zeros(len(points_found), dtype=int)
    for i in range(unique_pts.shape[0]):
        for j in range(len(points_found)):
            if np.allclose(unique_pts[i], points_found[j], equal_nan=True):
                vertex_candidates[j] = i
    vertex_candidates = vertex_candidates.reshape(numLines, numLines)


    # Find the best endpoints for each line (closest to their inliers)
    bestEndpoints = np.zeros((numLines, 2), dtype=int)
    for i in range(numLines):
        vertex_candidates_i = vertex_candidates[i]
        vertex_candidates_i = vertex_candidates_i[np.all(np.isfinite(unique_pts[vertex_candidates_i]), axis=1)]
        num_inliers = []
        endpoint_ids = []
        endpoint_distances = []
        # See how good of a fit every pair in the list is
        for point_pair in combinations(vertex_candidates_i, 2):
            p0 = unique_pts[point_pair[0]]
            p1 = unique_pts[point_pair[1]]
            d0 = np.linalg.norm(inliers[i] - p0, axis=1)
            d1 = np.linalg.norm(inliers[i] - p1, axis=1)
            ep = l[i][-1]
            p0_dist_ep0  = np.linalg.norm(p0 - np.array([ep[0][0], ep[1][0]]))
            p1_dist_ep0  = np.linalg.norm(p1 - np.array([ep[0][0], ep[1][0]]))
            p0_dist_ep1  = np.linalg.norm(p0 - np.array([ep[0][1], ep[1][1]]))
            p1_dist_ep1  = np.linalg.norm(p1 - np.array([ep[0][1], ep[1][1]]))
            num_inliers.append(np.sum(d0 <= inlierRadius) + np.sum(d1 <= inlierRadius))
            endpoint_ids.append(list(point_pair))
            endpoint_distances.append([p0_dist_ep0, p0_dist_ep1, p1_dist_ep0, p1_dist_ep1])

        iter = 1
        while np.sum(np.array(num_inliers) > 0) < 2 and iter < max_num_expansions:
            # Expand the search radius
            for k, point_pair in enumerate(combinations(vertex_candidates_i, 2)):
                p0 = unique_pts[point_pair[0]]
                p1 = unique_pts[point_pair[1]]
                d0 = np.linalg.norm(inliers[i] - p0, axis=1)
                d1 = np.linalg.norm(inliers[i] - p1, axis=1)
                num_inliers[k] = (np.sum(d0 <= radius_expansion_factor*iter*inlierRadius) 
                                    + np.sum(d1 <= radius_expansion_factor*iter*inlierRadius))
                p0_dist_ep0  = np.linalg.norm(p0 - np.array([ep[0][0], ep[1][0]]))
                p1_dist_ep0  = np.linalg.norm(p1 - np.array([ep[0][0], ep[1][0]]))
                p0_dist_ep1  = np.linalg.norm(p0 - np.array([ep[0][1], ep[1][1]]))
                p1_dist_ep1  = np.linalg.norm(p1 - np.array([ep[0][1], ep[1][1]]))
                endpoint_distances[k] = [p0_dist_ep0, p0_dist_ep1, p1_dist_ep0, p1_dist_ep1]
                iter += 1

        best_pair_idx = np.argmax(num_inliers)
        # If we're too far away from the original endpoints (computed during line fitting), then just use those
        # p0 = unique_pts[endpoint_ids[best_pair_idx][0]]
        # p1 = unique_pts[endpoint_ids[best_pair_idx][1]]
        # p0_ok = endpoint_distances[best_pair_idx][0] <= max_ep_dist or endpoint_distances[best_pair_idx][1] <= max_ep_dist
        # p1_ok = endpoint_distances[best_pair_idx][2] <= max_ep_dist or endpoint_distances[best_pair_idx][3] <= max_ep_dist
        # if not p0_ok and not p1_ok:
        #     # Neither points are good
        #     unique_pts = np.append(unique_pts, [[l[i][-1][0][0], l[i][-1][1][0]], [l[i][-1][0][1], l[i][-1][1][1]]], axis=0)
        #     endpoint_ids[best_pair_idx][0] = int(unique_pts.shape[0]) - 2
        #     endpoint_ids[best_pair_idx][1] = int(unique_pts.shape[0]) - 1
        # else:
        #     if not p1_ok:
        #         # Find which end point p2 is closer to, then choose the other one
        #         idx = int(endpoint_distances[best_pair_idx][2] <= max_ep_dist)
        #         unique_pts = np.append(unique_pts, [[l[i][-1][0][idx], l[i][-1][1][idx]]], axis=0)
        #         endpoint_ids[best_pair_idx][0] = int(unique_pts.shape[0]) - 1
        #     if not p1_ok:
        #         # Find which end point p1 is closer to, then choose the other one
        #         idx = int(endpoint_distances[best_pair_idx][0] <= max_ep_dist)
        #         unique_pts = np.append(unique_pts, [[l[i][-1][0][idx], l[i][-1][1][idx]]], axis=0)
        #         endpoint_ids[best_pair_idx][1] = int(unique_pts.shape[0]) - 1
            
        bestEndpoints[i] = endpoint_ids[best_pair_idx]
            

        
        
        # x_max = np.max(inliers[i][:, 0])
        # x_min = np.min(inliers[i][:, 0])
        # y_max = np.max(inliers[i][:, 1])
        # y_min = np.min(inliers[i][:, 1])
        # for j in range(numLines):
        #     x_max_j = np.max(inliers[j][:, 0])
        #     x_min_j = np.min(inliers[j][:, 0])
        #     y_max_j = np.max(inliers[j][:, 1])
        #     y_min_j = np.min(inliers[j][:, 1])
        #     if x_max_j > x_max:
        #         x_max = x_max_j
        #     if x_min_j < x_min:
        #         x_min = x_min_j
        #     if y_max_j > y_max:
        #         y_max = y_max_j
        #     if y_min_j < y_min:
        #         y_min = y_min_j
        #
        # fig, ax = plt.subplots()
        # ax.scatter(inliers[i][:, 0], inliers[i][:, 1], c='k')
        # cmap = mpl.colormaps["cool"]
        # max_inliers = float(np.max(num_inliers))
        #
        # # Plot every line
        # for j in range(numLines):
        #     x, y = GetLinePts(lineList[j], np.array([x_min, x_max]), np.array([y_min, y_max]))
        #     ax.plot(x, y, c='b', ls='-', label="Fitted Lines"*(i==1))
        # # Plot all the end points
        # for j in range(len(num_inliers)):
        #     c = cmap(num_inliers[j]/max_inliers)
        #     pt0 = unique_pts[endpoint_ids[j][0]]
        #     pt1 = unique_pts[endpoint_ids[j][1]]
        #     ax.scatter(pt0[0], pt0[1], c=c)
        #     ax.scatter(pt1[0], pt1[1], c=c)
        #
        #     circle0 = ptc.Circle(pt0, inlierRadius)
        #     circle1 = ptc.Circle(pt1, inlierRadius)
        #     circle0.set_fill(False)
        #     circle1.set_fill(False)
        #     ax.add_artist(circle0)
        #     ax.add_artist(circle1)
        # pad = 1
        # ax.set_xlim((x_min - pad, x_max + pad))
        # ax.set_ylim((y_min - pad, y_max + pad))
        # plt.show()


        # The two points with the most inliers will be the ends of this segment
        # minIdx = np.argpartition(numInliers, -2)[-2:]
        # bestEndpoints[i, 0] = liXlj[i, minIdx[0]]
        # bestEndpoints[i, 1] = liXlj[i, minIdx[1]]
    return bestEndpoints, unique_pts


def AdjustVertices(corners, pts, sigma):
    numLines = corners.shape[0]
    # Get the unique set of endpoints
    corners_2d = np.reshape(corners, (2*numLines, -1))
    corners_unique = np.unique(corners_2d, axis=0)
    



def MakeGrayscale(pts, scale=5):

    N = pts.shape[0]
    pts = pts*scale
    # Generate a grid for the image
    padding = 1*scale
    minX = np.min(pts[:, 0]) - padding
    maxX = np.max(pts[:, 1]) + padding
    minY = np.min(pts[:, 1]) - padding
    maxY = np.max(pts[:, 1]) + padding

    width = int(np.ceil(maxX - minX)) + 1
    height = int(np.ceil(maxY - minY)) + 1

    img = np.zeros((height, width), dtype=np.uint8)
    pts[:, 0] = pts[:, 0] - minX
    pts[:, 1] = pts[:, 1] - minY
    ptsRounded = np.round(pts).astype(int)
    img[ptsRounded[:, 1], ptsRounded[:, 0]] = 255
    return img




if __name__ == "__main__":
    truthPts, noisyPts = GenerateFloorplan(1, N=100, sigma=0.2)
    
    # border_points = np.load("border_points.npy")
    # noisyPts = border_points[border_points[:, -1] == 3, 0:2]

    # l, cost, inliers = FitLine_ransac(noisyPts, alpha=1, beta=1, thr_d=0.3)
    #
    # xFit = np.array([np.min(noisyPts[:, 0]), np.max(noisyPts[:, 0])])
    # yFit = -(l[0]*xFit + l[2])/l[1]
    #
    # fig, ax = plt.subplots()
    # ax.scatter(truthPts[:,0], truthPts[:,1], c='k')
    # ax.scatter(noisyPts[:,0], noisyPts[:,1], c='r')
    # ax.scatter(noisyPts[inliers, 0], noisyPts[inliers, 1], c='g')
    # ax.plot(xFit, yFit, c='g', ls='-')
    # pad = 1
    # ax.set_xlim((np.min(noisyPts[:, 0]) - pad, np.max(noisyPts[:, 0]) + pad))
    # ax.set_ylim((np.min(noisyPts[:, 1]) - pad, np.max(noisyPts[:, 1]) + pad))

    l, cost, inliers = FitMultipleLines(noisyPts, thr_d=0.4, beta=2)
    # l, cost, inliers = FitMultipleLines(noisyPts, thr_d=0.075, beta=2, minOutlierFrac=0.05)
    img_raw = MakeGrayscale(noisyPts)
    img = cv2.GaussianBlur(img_raw, (3, 3), 0)

    # #Create default parametrization LSD
    # lsd = cv2.createLineSegmentDetector(0)
    #
    # #Detect lines in the image
    # lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines
    #
    # #Draw detected lines in the image
    # drawn_img = lsd.drawSegments(img,lines)
    
    # dst = cv2.cornerHarris(img, 2, 27, 0.04)
    # dst = cv2.dilate(dst,None)
    # # Threshold for an optimal value, it may vary depending on the image.
    # img[dst>0.1*dst.max()]=255
    # cv2.imshow('dst',img)

    vertex_ids, vertex_points = FindVertices(l, inliers, inlierRadius=0.1, max_ep_dist=1.0)

    fig, ax = plt.subplots()
    # ax.scatter(truthPts[:,0], truthPts[:,1], c='k', label="Truth Points")
    ax.scatter(noisyPts[:,0], noisyPts[:,1], c='r', label="Noisy Points")
    for i in range(len(l)):
        x_lim = np.array([np.min(inliers[i][:, 0]), np.max(inliers[i][:, 0])])
        y_lim = np.array([np.min(inliers[i][:, 1]), np.max(inliers[i][:, 1])])
        x_line, y_line = GetLinePts(l[i], x_lim, y_lim)
        ax.scatter(inliers[i][:, 0], inliers[i][:, 1], c='g')
        ax.plot(x_line, y_line, c='b', ls='-', label="Fitted Lines"*(i==1))
        ax.scatter(vertex_points[vertex_ids[i, 0]][0], vertex_points[vertex_ids[i, 0]][1], c='b')
        ax.scatter(vertex_points[vertex_ids[i, 1]][0], vertex_points[vertex_ids[i, 1]][1], c='b')
        ax.scatter(l[i][-1][0], l[i][-1][1], marker="*")
        x_ep = [vertex_points[vertex_ids[i, 0]][0], vertex_points[vertex_ids[i, 1]][0]]
        y_ep = [vertex_points[vertex_ids[i, 0]][1], vertex_points[vertex_ids[i, 1]][1]]
        ax.plot(x_ep, y_ep, c='k', ls='--')
        pad = 1
        ax.set_xlim((np.min(noisyPts[:, 0]) - pad, np.max(noisyPts[:, 0]) + pad))
        ax.set_ylim((np.min(noisyPts[:, 1]) - pad, np.max(noisyPts[:, 1]) + pad))
        ax.annotate(f"l{i}", ((x_line[1] + x_line[0])/2, (y_line[1] + y_line[0])/2))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.axis("equal")

    # hull = ConvexHull(noisyPts)
    # fig2, ax2 = plt.subplots()
    # ax2.scatter(truthPts[:,0], truthPts[:,1], c='k', label="Truth Points")
    # ax2.scatter(noisyPts[:,0], noisyPts[:,1], c='r', label="Noisy Points")
    # for simplex in hull.simplices:
    #     plt.plot(noisyPts[simplex, 0], noisyPts[simplex, 1], 'k-')
    #
    # fig3, ax3 = plt.subplots()
    # ax3.imshow(img)



    plt.show()

