import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
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
    l_best = np.zeros(3)
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
        num_inliers = np.sum(np.abs(d) <= thr_d)


        # x_lim = np.array([np.min(pts[:, 0]), np.max(pts[:, 0])])
        # y_lim = np.array([np.min(pts[:, 1]), np.max(pts[:, 1])])
        # x, y = GetLinePts(l, x_lim, y_lim)
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


        U = CalculateUniformity(l, pts)
        obj_fun = alpha*num_inliers/N + beta*U
        if obj_fun > val_to_max:
            num_inliers_best = num_inliers
            val_to_max = obj_fun
            l_best = np.copy(l)
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


def FindVertices(lineList, inliers, inlierRadius=0.5):
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

    unique_pts = np.unique(points_found, axis=0)

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
        # See how good of a fit every pair in the list is
        for point_pair in combinations(vertex_candidates_i, 2):
            p0 = unique_pts[point_pair[0]]
            p1 = unique_pts[point_pair[1]]
            d0 = np.linalg.norm(inliers[i] - p0, axis=1)
            d1 = np.linalg.norm(inliers[i] - p1, axis=1)
            num_inliers.append(np.sum(d0 <= inlierRadius) + np.sum(d1 <= inlierRadius))
            endpoint_ids.append(point_pair)
        best_pair_idx = np.argmax(num_inliers)
        bestEndpoints[i] = endpoint_ids[best_pair_idx]

        
        
        # for j in range(i, numLines):
        #     if j != i:
        #         if np.all(np.isfinite(liXlj[i, j])):
        #             allInliers = np.vstack((inliers[i],  inliers[j]))
        #             # Calculate distance between intersection point and inliers for both lines
        #             d = np.linalg.norm(allInliers - liXlj[i, j], axis=1)
        #             numInliers[j] = np.sum(d <= inlierRadius)
        #             d_i = np.linalg.norm(inliers[i] - liXlj[i, j], axis=1)
        #             d_j = np.linalg.norm(inliers[j] - liXlj[i, j], axis=1)
        #             inliers_i = np.sum(d_i <= inlierRadius)/inliers[i].size
        #             inliers_j = np.sum(d_j <= inlierRadius)/inliers[j].size
        #             numInliers[j] = 0.5*inliers_i + 0.5*inliers_j
        #         else:
        #             numInliers[j] = -1

                # fig, ax = plt.subplots()
                # ax.scatter(allInliers[:, 0], allInliers[:, 1], c='k')
                # ax.scatter(liXlj[i, j, 0], liXlj[i, j, 1], c='g')
                #
                # xFit = np.array([np.min(allInliers[:, 0]), np.max(allInliers[:, 0])])
                # yFit = -(lineList[i][0]*xFit + lineList[i][2])/lineList[i][1]
                # yFit2 = -(lineList[j][0]*xFit + lineList[j][2])/lineList[j][1]
                # circle = ptc.Circle(liXlj[i,j], inlierRadius)
                # circle.set_fill(False)
                # ax.add_artist(circle)
                # ax.plot(xFit, yFit, c='b', ls='-')
                # ax.plot(xFit, yFit2, c='b', ls='-')
                # pad = 1
                # ax.set_xlim((np.min(allInliers[:, 0]) - pad, np.max(allInliers[:, 0]) + pad))
                # ax.set_ylim((np.min(allInliers[:, 1]) - pad, np.max(allInliers[:, 1]) + pad))
                # print(numInliers[j])
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
    truthPts, noisyPts = GenerateFloorplan(3, N=100, sigma=0.2)

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

    vertex_ids, vertex_points = FindVertices(l, inliers)

    fig, ax = plt.subplots()
    ax.scatter(truthPts[:,0], truthPts[:,1], c='k', label="Truth Points")
    ax.scatter(noisyPts[:,0], noisyPts[:,1], c='r', label="Noisy Points")
    for i in range(len(l)):
        x_lim = np.array([np.min(inliers[i][:, 0]), np.max(inliers[i][:, 0])])
        y_lim = np.array([np.min(inliers[i][:, 1]), np.max(inliers[i][:, 1])])
        x_line, y_line = GetLinePts(l[i], x_lim, y_lim)
        ax.scatter(inliers[i][:, 0], inliers[i][:, 1], c='g')
        ax.plot(x_line, y_line, c='b', ls='-', label="Fitted Lines"*(i==1))
        ax.scatter(vertex_points[vertex_ids[i, 0]][0], vertex_points[vertex_ids[i, 0]][1], c='b')
        ax.scatter(vertex_points[vertex_ids[i, 1]][0], vertex_points[vertex_ids[i, 1]][1], c='b')
        pad = 1
        ax.set_xlim((np.min(noisyPts[:, 0]) - pad, np.max(noisyPts[:, 0]) + pad))
        ax.set_ylim((np.min(noisyPts[:, 1]) - pad, np.max(noisyPts[:, 1]) + pad))
        ax.annotate(f"l{i}", ((x_line[1] + x_line[0])/2, (y_line[1] + y_line[0])/2))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

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

