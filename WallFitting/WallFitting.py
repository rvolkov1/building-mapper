import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cv2

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


def CalculateDistFromLine(l, x):
    # l (3,)
    # x (N,2)
    N = x.shape[0]
    l = l.reshape((3, 1))
    lNorm = np.linalg.norm(l[0:2])
    x_h = np.hstack((x, np.ones((N, 1))))
    d = x_h @ l / lNorm
    return d


def FitLine_ransac(pts, thr_d=0.8, maxIter=10000, alpha=1, beta=1):
    N = pts.shape[0]
    idxList = np.arange(N)
    lBest = np.zeros(3)
    numInliersBest = 0
    costBest = 0
    inliersBest = np.zeros(N)

    rng = np.random.default_rng()

    iter = 0
    while numInliersBest < N and iter < maxIter:
        # Select two points
        idxChoice = rng.choice(idxList, 2, replace=False)
        pt1 = pts[idxChoice[0]]
        pt2 = pts[idxChoice[1]]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        l = np.array([dy, -dx, dx*pt1[1] - dy*pt1[0]])
        d = CalculateDistFromLine(l, pts).flatten()
        numInliers = np.sum(np.abs(d) <= thr_d)
        # if numInliers > numInliersBest:
        #     numInliersBest = numInliersBest
        #     lBest = np.copy(l)
        U = CalculateUniformity(l, pts)
        cost = alpha*numInliers/N + beta*U
        if cost > costBest:
            costBest = cost
            lBest = np.copy(l)
            inliersBest = np.abs(d) <= thr_d

        iter += 1

    return lBest, costBest, inliersBest
        
def CalculateUniformity(l, pts, thr_d=0.8, showOutput=False):
    # Check for how the inliers are distributed along the line l
    d = CalculateDistFromLine(l, pts)
    inlierSelection = (np.abs(d) <= thr_d).flatten()
    ptsInliers = pts[inlierSelection, :]
    lineAngle = np.atan2(l[1], l[0])
    rotationAngle = lineAngle - np.pi/2
    ct = np.cos(rotationAngle)
    st = np.sin(rotationAngle)
    DCM = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])
    ptsRotated = (DCM[0:2, 0:2] @ ptsInliers.T).T
    lRotated = DCM @ l

    # Shift to center of mass
    numInliers = ptsInliers.shape[0]
    x_c = np.sum(ptsRotated[:, 0])/numInliers
    y_c = np.sum(ptsRotated[:, 1])/numInliers
    ptsRotated[:, 0] -= x_c
    ptsRotated[:, 1] -= y_c
    lRotated[-1] = 0
    

    # Calculate the moment of inertia
    Ixx = np.sum(ptsRotated[:, 1]**2)/numInliers
    Iyy = np.sum(ptsRotated[:, 0]**2)/numInliers
    Ixy = np.sum(ptsRotated[:, 0]*ptsRotated[:, 1])/numInliers

    # Uniformity score (kind of like pearson test statistic)
    nBins = 10
    h = np.histogram(ptsRotated[:, 0], bins=nBins)[0]
    E = numInliers/nBins
    U = 1 - np.sum((h - E)**2)/(numInliers - E)**2

    if showOutput == True:
        momentText = f"Ixx = {Ixx}, Iyy = {Iyy}, Ixy = {Ixy}"
        print(momentText)
        print(f"U = {U}")

        xFit = np.array([np.min(ptsInliers[:, 0]), np.max(ptsInliers[:, 0])])
        yFit = -(l[0]*xFit + l[2])/l[1]
        xFitRotated = np.array([np.min(ptsRotated[:, 0]), np.max(ptsRotated[:, 0])])
        yFitRotated = -(lRotated[0]*xFitRotated + lRotated[2])/lRotated[1]
        fig, ax = plt.subplots(1, 2)
        ax[0].scatter(pts[:, 0], pts[:, 1], c="r")
        ax[0].scatter(ptsInliers[:, 0], ptsInliers[:, 1], c="g")
        ax[0].plot(xFit, yFit, c="g", ls="-")
        ax[1].scatter(ptsRotated[:, 0], ptsRotated[:, 1], c="g")
        ax[1].plot(xFitRotated, yFitRotated, c='g', ls='-')
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
    cost = []

    while outlierFrac > minOutlierFrac:
        l_i, cost_i, inliers_i = FitLine_ransac(ptsAvail, thr_d=thr_d, alpha=alpha, beta=beta)
        inlierPts.append(ptsAvail[inliers_i])
        inliers.append(inliers_i)
        ptsAvail = np.copy(ptsAvail[np.logical_not(inliers_i)])
        l.append(l_i)
        cost.append(cost_i)
        # Calculate fraction of inliers
        numInliers = 0
        for k in range(len(inliers)):
            numInliers += np.sum(inliers[k])
        outlierFrac = (N - numInliers)/N

        print(outlierFrac)
    
    return l, cost, inlierPts


def FindVertices(lineList, inliers, inlierRadius=1):
    # Each line should have two end points, therefore we look for two "real" intersections between each line element.
    # There would have to be some consistency check, as lines that don't correspond to corners may intersect at further away points.
    
    # Compute intersections
    numLines = len(lineList)
    liXlj = np.zeros((numLines, numLines, 2))
    for i, li in enumerate(lineList):
        for j, lj in enumerate(lineList):
            p = np.linalg.cross(li, lj)
            liXlj[i, j] = p[0:2]/(p[-1] + 1e-6)

    # Find the best endpoints for each line (closest to their inliers)
    bestEndpoints = np.zeros((numLines, 2, 2))
    for i in range(numLines):
        for j in range(i, numLines):
            numInliers = np.zeros(numLines)
            if j != i:
                allInliers = np.vstack((inliers[i],  inliers[j]))
                # Calculate distance between intersection point and inliers for both lines
                d = np.linalg.norm(allInliers - liXlj[i, j], axis=1)
                numInliers[j] = np.sum(d <= inlierRadius)
            # The two points with the most inliers will be the ends of this segment
            minIdx = np.argpartition(numInliers, -2)[-2:]
            bestEndpoints[i, 0] = liXlj[i, minIdx[0]]
            bestEndpoints[i, 1] = liXlj[i, minIdx[1]]
    return bestEndpoints



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

    l, cost, inliers = FitMultipleLines(noisyPts, thr_d=0.4)
    img_raw = MakeGrayscale(noisyPts)
    img = cv2.GaussianBlur(img_raw, (9, 9), 0)
    
    # dst = cv2.cornerHarris(img, 2, 27, 0.04)
    # dst = cv2.dilate(dst,None)
    # # Threshold for an optimal value, it may vary depending on the image.
    # img[dst>0.1*dst.max()]=255
    # cv2.imshow('dst',img)

    vertices = FindVertices(l, inliers)

    fig, ax = plt.subplots()
    ax.scatter(truthPts[:,0], truthPts[:,1], c='k', label="Truth Points")
    ax.scatter(noisyPts[:,0], noisyPts[:,1], c='r', label="Noisy Points")
    for i in range(len(l)):
        xFit = np.array([np.min(inliers[i][:, 0]), np.max(inliers[i][:, 0])])
        yFit = -(l[i][0]*xFit + l[i][2])/l[i][1]
        ax.scatter(inliers[i][:, 0], inliers[i][:, 1], c='g')
        ax.plot(xFit, yFit, c='b', ls='-', label="Fitted Lines"*(i==1))
        ax.scatter(vertices[i, 0, 0], vertices[i, 0, 1], c='b')
        ax.scatter(vertices[i, 1, 0], vertices[i, 1, 1], c='b')
        pad = 1
        ax.set_xlim((np.min(noisyPts[:, 0]) - pad, np.max(noisyPts[:, 0]) + pad))
        ax.set_ylim((np.min(noisyPts[:, 1]) - pad, np.max(noisyPts[:, 1]) + pad))
        ax.annotate(f"l{i}", ((xFit[1] + xFit[0])/2, (yFit[1] + yFit[0])/2))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    hull = ConvexHull(noisyPts)
    fig2, ax2 = plt.subplots()
    ax2.scatter(truthPts[:,0], truthPts[:,1], c='k', label="Truth Points")
    ax2.scatter(noisyPts[:,0], noisyPts[:,1], c='r', label="Noisy Points")
    for simplex in hull.simplices:
        plt.plot(noisyPts[simplex, 0], noisyPts[simplex, 1], 'k-')

    fig3, ax3 = plt.subplots()
    ax3.imshow(img)



    plt.show()

