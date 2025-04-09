import numpy as np
import matplotlib.pyplot as plt

def GenerateFloorplan(N=10, sigma=1, rngSeed=None):
    corner1 = [0, 0]
    corner2 = [10, 0]
    corner3 = [10, 10]
    corner4 = [0, 10]
    endpoints = [corner1, corner2, corner3, corner4, corner1]
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


def FitLine_ransac(pts, thr_d=0.8, maxIter=10000):
    N = pts.shape[0]
    idxList = np.arange(N)
    lBest = np.zeros(3)
    dBestNorm = np.inf
    numInliersBest = 0

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
        d = CalculateDistFromLine(l, pts)
        numInliers = np.sum(np.abs(d) <= thr_d)
        if numInliers > numInliersBest:
            numInliersBest = numInliersBest
            lBest = np.copy(l)

        iter += 1

    return lBest, numInliersBest
        
def GapCheck(l, pts, thr_d=0.8):
    # Check for how the inliers are distributed along the line l
    d = CalculateDistFromLine(l, pts)
    inlierSelection = (np.abs(d) <= thr_d).flatten()
    ptsInliers = pts[inlierSelection, :]
    lineAngle = np.atan2(l[1], l[0])
    rotationAngle = lineAngle - np.pi/2
    ct = np.cos(rotationAngle)
    st = np.sin(rotationAngle)
    R = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])
    ptsRotated = (R[0:2, 0:2] @ ptsInliers.T).T
    lRotated = R @ l

    # Calculate the moment of inertia
    Ixx = np.sum(ptsRotated[:, 1]**2)
    Ixy = np.sum(ptsRotated[:, 0]*ptsRotated[:, 1])
    momentText = f"Ixx = {Ixx}, Ixy = {Ixy}"

    xFit = np.array([np.min(ptsInliers[:, 0]), np.max(ptsInliers[:, 0])])
    yFit = -(l[0]*xFit + l[2])/l[1]
    xFitRotated = np.array([np.min(ptsRotated[:, 0]), np.max(ptsRotated[:, 0])])
    yFitRotated = -(lRotated[0]*xFitRotated + lRotated[2])/lRotated[1]
    fig, ax = plt.subplots(1, 3)
    ax[0].scatter(pts[:, 0], pts[:, 1], c="r")
    ax[0].scatter(ptsInliers[:, 0], ptsInliers[:, 1], c="g")
    ax[0].plot(xFit, yFit, c="g", ls="-")
    ax[1].scatter(ptsRotated[:, 0], ptsRotated[:, 1], c="g")
    ax[1].plot(xFitRotated, yFitRotated, c='g', ls='-')
    ax[2].hist(ptsRotated[:, 0])
    ax[0].text(1, 1, momentText)

    plt.show()


if __name__ == "__main__":
    truthPts, noisyPts = GenerateFloorplan(N=100, sigma=0.2)

    lBest, dBestNorm = FitLine_ransac(noisyPts)

    GapCheck(lBest, noisyPts)

    xFit = np.array([np.min(noisyPts[:, 0]), np.max(noisyPts[:, 0])])
    yFit = -(lBest[0]*xFit + lBest[2])/lBest[1]
    print(xFit)
    print(yFit)

    fig, ax = plt.subplots()
    ax.scatter(truthPts[:,0], truthPts[:,1], c='k')
    ax.scatter(noisyPts[:,0], noisyPts[:,1], c='r')
    ax.plot(xFit, yFit, c='g', ls='-')


    plt.show()

