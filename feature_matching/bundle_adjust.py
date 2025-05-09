import numpy as np
from scipy.sparse import lil_matrix, block_diag, csc_matrix
import scipy.sparse.linalg as sps
import time
from scipy.optimize import least_squares

def skew(v):
    v = v.flatten()
    cross = np.array([[0, -v[2], v[1]], 
                      [v[2], 0, -v[0]], 
                      [-v[1], v[0], 0]])
    return cross

def dcm2quat(R):
    eta = np.sqrt(1 + np.linalg.trace(R))/2
    eps = np.zeros(3)

    if (eta != 0):
        eps[0] = (R[1,2] - R[2,1])/4/eta;
        eps[1] = (R[2,0] - R[0,2])/4/eta;
        eps[2] = (R[0,1] - R[1,0])/4/eta;
    else:
        eps[0] = np.sqrt((1 + R[0, 0] + 1)/2)
        eps[1] = np.sqrt((1 + R[1, 1] + 1)/2)
        eps[2] = np.sqrt((1 + R[2, 2] + 1)/2)

        if eps[0] > 0:
          eps[1] = np.sign(R[0, 1])*eps[1]
          eps[2] = np.sign(R[0, 2])*eps[2]
        elif eps[1] > 0:
          eps[0] = np.sign(R[0, 1])*eps[0]
          eps[2] = np.sign(R[1, 2])*eps[2]
        else:
          eps[0] = np.sign(R[0, 2])*eps[0]
          eps[1] = np.sign(R[1, 2])*eps[1]
    return np.concatenate(([eta], eps)).reshape((4, 1))

def quat2dcm(q):
    q = q.flatten()
    eps = q[1:].reshape((3, 1))
    eta = q[0]

    I = np.eye(3)
    R = (2*eta**2 - 1)*I + 2*(eps @ eps.T) - 2*eta*skew(eps)
    return R
    

def ParseDataFile(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse the first line
    nCams, nPoints, nObs = map(int, lines[0].split())
    
    # Parse the observations: next nObs lines
    obs_lines = lines[1:1+nObs]
    pts_obs = np.array([list(map(float, line.split())) for line in obs_lines])
    
    # Parse camera parameters: next 9*nCams lines
    cam_start = 1 + nObs
    cam_end = cam_start + 9 * nCams
    cam_lines = lines[cam_start:cam_end]
    cam = np.array([float(line.strip()) for line in cam_lines]).reshape(9, nCams, order='F')
    
    # Parse points: remaining lines
    point_lines = lines[cam_end:]
    pts_3d = np.array([float(line.strip()) for line in point_lines])
    
    return pts_obs, cam, pts_3d

def FormCameraMatrix(camRaw):
    nCams = camRaw.shape[1]
    cam = []
    camParams = np.zeros(nCams*7)
    for camID in range(nCams):
        p = camRaw[0:3, camID]
        theta = np.linalg.norm(p)
        if abs(theta) > 1e-6:
            w = p/theta
        else:
            w = p*0
        I = np.eye(3)
        w_cross = skew(w)

        # Calculate DCM
        R = I + np.sin(theta)*w_cross + (1 - np.cos(theta))*(w_cross @ w_cross)
        if camID == 0:
            print(R)
        q = dcm2quat(R).flatten()
        t = camRaw[3:6, camID]
        camParams[camID*7:camID*7+4] = q
        camParams[camID*7+4:camID*7+7] = t
        f = camRaw[6, camID]
        k1 = camRaw[7, camID]
        k2 = camRaw[8, camID]
        cam.append([f, k1, k2])
            
    return cam, camParams


def ComputeReprojError(params, cam, nPts, pts_obs):
    # Description: Computes the total reprojection error over all points
    # Inputs:
    #   var     | type    | size      | descr
    #   cam     | list    | nCams     | A list containing the camera matrices (cam[i][0] = K,  cam[i][1] = [R_i | t_i]). Each index corresponds to the camera ID
    #   pts_3d  | ndarray | (nPts, 3) | The estimated 3d points in the World frame. Each row index corresponds to the point ID
    #   pts_obs | ndarray | (nObs, 4) | The observed points in each camera frame. Column order = (camID, ptID, u, v)
    nCams = len(cam)
    totalReprojError = 0
    pts_3d = np.reshape(params[nCams*7:], (nPts, 3))
    for camID in range(nCams):
        # M = cam[camID][0] @ cam[camID][1]
        # Get all the observed points in the camera frame
        pointsInCam = pts_obs[pts_obs[:, 0] == camID, 1:]
        # If we assume all point IDs in pts_obs are in ascending order, then we can just grab the 3d points and not worry about order
        X_w = np.vstack((pts_3d[pointsInCam[:, 0].astype(int)].T, np.ones((1, pointsInCam.shape[0]))))

        # Estimated projected points
        # Compute camera matrix
        q = params[camID*7:camID*7+4]
        R = quat2dcm(q)
        t = params[camID*7+4:camID*7+7]
        M = np.block([R, t.reshape((3, 1))])
        x_homo = M @ X_w
        x_homo = -x_homo[0:2, :]/x_homo[-1, :]


        # x_hat = cam[camID][0] @ x_homo
        f = cam[camID][0]
        k1 = cam[camID][1]
        k2 = cam[camID][2]
        p_norm = np.linalg.norm(x_homo, axis=0)
        r = 1 + k1 * p_norm**2 + k2 * p_norm**4
        x_hat = f*r*x_homo

        e = pointsInCam[:, 1:] - x_hat.T
        # Add the sum of the 2-norm to the total reprojection error
        totalReprojError += np.sum(np.linalg.norm(e, axis=1))
    return totalReprojError

def ComputeResidual(params, cam, nPts, pts_obs):
    # Description: Computes the total reprojection error over all points
    # Inputs:
    #   var     | type    | size      | descr
    #   cam     | list    | nCams     | A list containing the camera matrices (cam[i][0] = K,  cam[i][1] = [R_i | t_i]). Each index corresponds to the camera ID
    #   pts_3d  | ndarray | (nPts, 3) | The estimated 3d points in the World frame. Each row index corresponds to the point ID
    #   pts_obs | ndarray | (nObs, 4) | The observed points in each camera frame. Column order = (camID, ptID, u, v)
    nCams = len(cam)
    nObs = pts_obs.shape[0]
    pts_3d = np.reshape(params[nCams*7:], (nPts, 3))
    e = np.zeros((2*nObs, 1))
    for obs in range(nObs):
        camID = pts_obs[obs, 0].astype(int)
        ptId = pts_obs[obs, 1].astype(int)
        u, v = pts_obs[obs, 2:]
        X_w = np.concatenate((pts_3d[ptId].reshape((3, 1)), np.ones((1,1))), axis=0)
        # Estimated projected points
        # Compute camera matrix
        q = params[camID*7:camID*7+4]
        R = quat2dcm(q)
        t = params[camID*7+4:camID*7+7]
        M = np.block([R, t.reshape((3, 1))])
        x_homo = M @ X_w
        x_homo = -x_homo[0:2, :]/x_homo[-1, :]
        f = cam[camID][0]
        k1 = cam[camID][1]
        k2 = cam[camID][2]
        p_norm = np.linalg.norm(x_homo, axis=0)
        r = 1 + k1 * p_norm**2 + k2 * p_norm**4
        x_hat = f*r*x_homo
        e[2*obs:2*obs+2, 0] = (np.array([[u], [v]]) - x_hat).flatten()

    return e

def ComputeJacobian_basic(params, cam, nPts, pts_obs):
    nCams = len(cam)
    nObs = pts_obs.shape[0]
    pts_3d = np.reshape(params[nCams*7:], (nPts, 3))
    J = np.zeros((nObs*2, 7*nCams + nPts*3))
    # Build the jacobian for each observation
    for row in range(nObs):
        # print(row/nObs)
        camID = pts_obs[row, 0].astype(int)
        ptId = pts_obs[row, 1].astype(int)
        f = cam[camID][0]
        q = params[camID*7:camID*7+4]
        R = quat2dcm(q)
        t = params[camID*7+4:camID*7+7]
        M = np.block([R, t.reshape((3, 1))])
        # dp_hat/dp_tilde
        K = np.array([[f, 0, 0], [0, f, 0]])
        # dp_tilde/dX'
        X_w = np.array([pts_3d[ptId, 0], pts_3d[ptId, 1], pts_3d[ptId, 2], 0])
        X_prime = M @ X_w.reshape((4, 1))
        xp = X_prime[0, 0]
        yp = X_prime[1, 0]
        zp = X_prime[2, 0]
        dp_dxprime = np.array([[ 1/zp, 0, -xp/zp ], [ 0, 1/zp, -yp/zp ], [0, 0, 0]])
        # dX'/dX
        # For the camera params
        qw, qx, qy, qz = q.flatten()

        # dX'/dR
        O = np.zeros((1, 3))
        dXp_dR = np.block([[X_prime.T, O, O], [O, X_prime.T, O], [O, O, X_prime.T]])
        dR_dq = np.array([[0, 0, -4*qy, -4*qz], 
                          [-2*qz, 2*qy, 2*qx, -2*qw], 
                          [2*qy, 2*qz, 2*qw, 2*qx], 
                          [2*qz, 2*qy, 2*qx, 2*qw], 
                          [0, -4*qx, 0, -4*qz], 
                          [-2*qx, -2*qw, 2*qz, 2*qy], 
                          [-2*qy, 2*qz, -2*qw, 2*qx], 
                          [2*qx, 2*qw, 2*qz, 2*qy], 
                          [0, -4*qx, -4*qy, 0]])
        dXp_dt = np.eye(3)

        J_c1 = K @ dp_dxprime @ dXp_dR @ dR_dq
        J_c2 = K @ dp_dxprime @ dXp_dt
        J_c = np.block([J_c1, J_c2])
        J_p = K @ dp_dxprime @ R

        J[row:row+2, camID*7:camID*7+7] = np.copy(J_c)
        J[row:row+2, 7*nCams+ptId*3:7*nCams+ptId*3+3] = np.copy(J_p)

    return J


def ComputeJacobian_sparse(params, cam, nPts, pts_obs):
    nCams = len(cam)
    nObs = pts_obs.shape[0]
    pts_3d = np.reshape(params[nCams*7:], (nPts, 3))
    # J = np.zeros((nObs*2, 7*nCams + nPts*3))

    # Let's first tackle the block diagonal matrix of point jacobians
    uniquePtIds, counts = np.unique(pts_obs[:, 1], return_counts=True)

    # Build point jacobians (block)
    nBlocks = pts_3d.shape[0]
    J_x = []
    D_inv = []
    for j in range(nBlocks):
        camInBlock = (pts_obs[pts_obs[:, 1] == j, 0]).flatten().astype(int)
        X_w = np.array([pts_3d[j, 0], pts_3d[j, 1], pts_3d[j, 2], 1])
        J_block = np.zeros((camInBlock.size*2, 3))
        for i in range(camInBlock.size):
            camId = camInBlock[i]
            f = cam[camId][0]
            K = np.array([[f, 0, 0], [0, f, 0]])
            q = params[camId*7:camId*7+4]
            R = quat2dcm(q)
            t = params[camId*7+4:camId*7+7]
            M = np.block([R, t.reshape((3, 1))])
            X_prime = M @ X_w.reshape((4, 1))
            xp = X_prime[0, 0]
            yp = X_prime[1, 0]
            zp = X_prime[2, 0]
            dp_dX_prime = np.array([[1/zp, 0, -xp/zp**2], [0, 1/zp, -yp/zp**2], [0, 0, 0]])
            J_block[2*i:2*i+2, :] = K @ dp_dX_prime @ R
        D_inv_block = np.linalg.inv(J_block.T @ J_block)
        J_x.append(J_block)
        D_inv.append(D_inv_block)
    J_x = block_diag(J_x, "csc")
    D_inv = block_diag(D_inv, "csc")

    J_c = np.zeros((2*nObs, nCams*7))
    # Build camera jacobians
    for row in range(nObs):
        camID = pts_obs[row, 0].astype(int)
        ptId = pts_obs[row, 1].astype(int)
        f = cam[camID][0]
        q = params[camID*7:camID*7+4]
        R = quat2dcm(q)
        t = params[camID*7+4:camID*7+7]
        M = np.block([R, t.reshape((3, 1))])
        # dp_hat/dp_tilde
        K = np.array([[f, 0, 0], [0, f, 0]])
        # dp_tilde/dX'
        X_w = np.array([pts_3d[ptId, 0], pts_3d[ptId, 1], pts_3d[ptId, 2], 0])
        X_prime = M @ X_w.reshape((4, 1))
        xp = X_prime[0, 0]
        yp = X_prime[1, 0]
        zp = X_prime[2, 0]
        dp_dxprime = np.array([[ 1/zp, 0, -xp/zp ], [ 0, 1/zp, -yp/zp ], [0, 0, 0]])
        # For the camera params
        qw, qx, qy, qz = q.flatten()
        # dX'/dR
        O = np.zeros((1, 3))
        dXp_dR = np.block([[X_prime.T, O, O], [O, X_prime.T, O], [O, O, X_prime.T]])
        dR_dq = np.array([[0, 0, -4*qy, -4*qz], 
                          [-2*qz, 2*qy, 2*qx, -2*qw], 
                          [2*qy, 2*qz, 2*qw, 2*qx], 
                          [2*qz, 2*qy, 2*qx, 2*qw], 
                          [0, -4*qx, 0, -4*qz], 
                          [-2*qx, -2*qw, 2*qz, 2*qy], 
                          [-2*qy, 2*qz, -2*qw, 2*qx], 
                          [2*qx, 2*qw, 2*qz, 2*qy], 
                          [0, -4*qx, -4*qy, 0]])
        dXp_dt = np.eye(3)
        J_c1 = K @ dp_dxprime @ dXp_dR @ dR_dq
        J_c2 = K @ dp_dxprime @ dXp_dt
        J_c[2*row:2*row+2, camID*7:camID*7+7] = np.block([J_c1, J_c2])

    return J_c, J_x, D_inv
    
def GaussNewton_sparse(res_fun, J_fun, params, cam, nPts, pts_obs, tol=1e-4):
    # Compute the initial cost
    res = res_fun(params, cam, nPts, pts_obs)
    cost = np.linalg.norm(res)
    print(cost)
    nCams = len(cam)
    while cost > tol:
        res = res_fun(params, cam, nPts, pts_obs)
        cost = np.linalg.norm(res)
        print(cost)
        J_c, J_x, D_inv = ComputeJacobian_sparse(params, cam, nPts, pts_obs)
        # D = J_x.T @ J_x
        A = J_c.T @ J_c
        B = J_c.T @ J_x
        # D_inv = sps.inv(D)
        
        e_c = J_c.T @ res
        e_x = J_x.T @ res
        delta_c = np.linalg.inv(A - B@D_inv@B.T) @ (e_c - B@D_inv@e_x)
        delta_x = D_inv @ (e_x - B.T@delta_c)

        params[0:7*nCams] += delta_c.flatten()
        params[7*nCams:] += delta_x.flatten()
        
    return params, res



if __name__ == "__main__":
    import sys
    # Use some of the Agarwal data to test. Can compare with Ceres
    if len(sys.argv) != 2:
        sys.exit("One argument for the path to the data file is required.")
    dataPath = sys.argv[1]

    pts_obs, camRaw, pts_3d = ParseDataFile(dataPath)
    cam, camParams = FormCameraMatrix(camRaw)
    params = np.concatenate((camParams, pts_3d))
    nPts = int(pts_3d.shape[0]/3)
    totalReprojError = ComputeReprojError(params, cam, nPts, pts_obs)
    print(totalReprojError)

    # J = ComputeJacobian(params, cam, nPts, pts_obs)

    t0 = time.time()
    params, res = GaussNewton_sparse(ComputeResidual, ComputeJacobian_sparse, params, cam, nPts, pts_obs)
    t1 = time.time()

    print("done")

