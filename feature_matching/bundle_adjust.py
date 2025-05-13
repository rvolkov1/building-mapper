from rotation_utils import skew, dcm2quat, quat2dcm
from numerical_jacobian import *
import numpy as np
from scipy.sparse import lil_matrix, block_diag, csc_matrix
import scipy.sparse.linalg as sps
import time
from scipy.optimize import least_squares


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


def ComputeReprojError(params, cam, num_points, pts_obs):
    """
    Compute the total reprojection error of a set of observations and points.

    Args:
        param (ndarray): Array containing parameters that are optimized over. 
            Shape = (num_cams*7 + num_points*3,)
        cam (list): A list containing camera focal length and distortion 
            parameters. Each index of the list is the camera ID. Each entry is 
            list with entries [f, k1, k2].
        num_points (int): The number of 3D points being estimated
        pts_obs (ndarray): Array containing the observed points in each image 
            along with the camera ID and point ID of the observation. The order
            of each entry is [cam_ID, point_ID, u, v]. Shape = (num_obs, 4).
    Returns:
        float: The total reprojection error
    """

    num_cams = len(cam)
    total_reproj_error = 0
    pts_3d = np.reshape(params[num_cams*7:], (num_points, 3))
    # Compute the reprojection error for each camera
    for cam_ID in range(num_cams):
        points_in_cam = pts_obs[pts_obs[:, 0] == cam_ID, 1:]
        # If we assume all point IDs in pts_obs are in ascending order, then we
        # can just grab the 3d points and not worry about order
        X_w = np.vstack((pts_3d[points_in_cam[:, 0].astype(int)].T, 
                         np.ones((1, points_in_cam.shape[0]))))

        # Estimated projected points
        # Compute camera matrix
        q = params[cam_ID*7:cam_ID*7+4]
        R = quat2dcm(q)
        t = params[cam_ID*7+4:cam_ID*7+7]
        M = np.block([R, t.reshape((3, 1))])
        x_tilde = M @ X_w
        x_tilde = -x_tilde[0:2, :]/x_tilde[-1, :]


        # x_hat = cam[camID][0] @ x_homo
        f = cam[cam_ID][0]
        k1 = cam[cam_ID][1]
        k2 = cam[cam_ID][2]
        p_norm = np.linalg.norm(x_tilde, axis=0)
        r = 1 + k1 * p_norm**2 + k2 * p_norm**4
        x_hat = f*r*x_tilde

        e = points_in_cam[:, 1:] - x_hat.T
        # Add the sum of the 2-norm to the total reprojection error
        total_reproj_error += np.sum(np.linalg.norm(e, axis=1))
    return total_reproj_error


def ComputeResidual(params, cam, num_points, pts_obs):
    """
    Compute the reprojection for each observation.

    Args:
        param (ndarray): Array containing parameters that are optimized over. 
            Shape = (num_cams*7 + num_points*3,)
        cam (list): A list containing camera focal length and distortion 
            parameters. Each index of the list is the camera ID. Each entry is 
            list with entries [f, k1, k2].
        num_points (int): The number of 3D points being estimated
        pts_obs (ndarray): Array containing the observed points in each image 
            along with the camera ID and point ID of the observation. The order
            of each entry is [cam_ID, point_ID, u, v]. Shape = (num_obs, 4).
    Returns:
        float: The residual for each observation
    """
    nCams = len(cam)
    nObs = pts_obs.shape[0]
    pts_3d = np.reshape(params[nCams*7:], (num_points, 3))
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

def ComputeJacobian_sparse(params, cam, nPts, pts_obs, D_damping_factor=0.1):
    nCams = len(cam)
    nObs = pts_obs.shape[0]
    pts_3d = np.reshape(params[nCams*7:], (nPts, 3))

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
            if q[0] < 0:
                q = -q
            q = q/np.linalg.norm(q)

            R = quat2dcm(q)
            t = params[camId*7+4:camId*7+7]
            M = np.block([R, t.reshape((3, 1))])
            X_prime = M @ X_w.reshape((4, 1))
            xp = X_prime[0, 0]
            yp = X_prime[1, 0]
            zp = X_prime[2, 0]
            dp_dX_prime = -np.array([[1/zp, 0, -xp/zp**2], [0, 1/zp, -yp/zp**2], [0, 0, 0]])
            J_block[2*i:2*i+2, :] = K @ dp_dX_prime @ R
        n = J_block.shape[1]
        D_inv_block = np.linalg.inv(J_block.T @ J_block + D_damping_factor*np.eye(n))
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
        if q[0] < 0:
            q = -q
        q = q/np.linalg.norm(q)
        
        R = quat2dcm(q)
        t = params[camID*7+4:camID*7+7]
        M = np.block([R, t.reshape((3, 1))])
        # dp_hat/dp_tilde
        K = np.array([[f, 0, 0], [0, f, 0]])
        # dp_tilde/dX'
        X_w = np.array([pts_3d[ptId, 0], pts_3d[ptId, 1], pts_3d[ptId, 2], 1])
        X_prime = M @ X_w.reshape((4, 1))
        xp = X_prime[0, 0]
        yp = X_prime[1, 0]
        zp = X_prime[2, 0]
        dp_dxprime = -np.array([[ 1/zp, 0, -xp/zp**2 ], [ 0, 1/zp, -yp/zp**2 ], [0, 0, 0]])
        # For the camera params
        qw, qx, qy, qz = q.flatten()
        # dX'/dR
        # O = np.zeros((1, 3))
        # dXp_dR = np.block([[X_prime.T, O, O], [O, X_prime.T, O], [O, O, X_prime.T]])
        # dR_dq = np.array([[0, 0, -4*qy, -4*qz], 
        #                   [-2*qz, 2*qy, 2*qx, -2*qw], 
        #                   [2*qy, 2*qz, 2*qw, 2*qx], 
        #                   [2*qz, 2*qy, 2*qx, 2*qw], 
        #                   [0, -4*qx, 0, -4*qz], 
        #                   [-2*qx, -2*qw, 2*qz, 2*qy], 
        #                   [-2*qy, 2*qz, -2*qw, 2*qx], 
        #                   [2*qx, 2*qw, 2*qz, 2*qy], 
        #                   [0, -4*qx, -4*qy, 0]])

        # v = np.array([qx, qy, qz]).reshape(3, 1)
        # a = np.array([xp, yp, zp]).reshape(3, 1)
        # dXp_dq2 = 2*np.block([qw*a + skew(v)@a, v.T@a*np.eye(3) + v@a.T - a@v.T - qw*skew(a)])
        # dXp_dq_numerical = GetJacobian_X_prime_q(params[camID*7:camID*7+7], X_w[0:3])

        dXp_dt = np.eye(3)
        # J_c1 = K @ dp_dxprime @ dXp_dR @ dR_dq
        # Just tyring this
        # J_c1[:, 1:] = -J_c1[:, 1:]
        # J_c1 = K @ dp_dxprime @ dXp_dq_numerical
        # J_c1 = K @ dp_dxprime @ dXp_dq2
        J_c1 = GetJacobian_full_q(params[camID*7:camID*7+7], X_w[0:3], f)[0:2, :]
        J_c2 = K @ dp_dxprime @ dXp_dt
        J_c[2*row:2*row+2, camID*7:camID*7+7] = np.block([J_c1, J_c2])

    return J_c, J_x, D_inv
    
def GaussNewton_sparse(res_fun, J_fun, params, cam, nPts, pts_obs, tol=1e-4, damping_factor=0.1):
    # Compute the initial cost
    res = res_fun(params, cam, nPts, pts_obs)
    reproj_error = np.linalg.norm(res)
    # I know this is a mess.
    nCams = len(cam)
    reproj_old = 1e5
    reproj_change = reproj_error
    res_list = []
    while abs(reproj_change) > tol:
        res = res_fun(params, cam, nPts, pts_obs)
        reproj_error = np.sum(np.linalg.norm(res.reshape(2, -1, order="f"), axis=0))
        res_list.append(reproj_error)
        reproj_change = reproj_error - reproj_old
        reproj_old = reproj_error
        print(reproj_error)
        J_c, J_x, D_inv = ComputeJacobian_sparse(params, cam, nPts, pts_obs, damping_factor)
        # D = J_x.T @ J_x
        A = J_c.T @ J_c + damping_factor*np.eye(J_c.shape[1])
        B = J_c.T @ J_x
        # D_inv = sps.inv(D)
        
        e_c = J_c.T @ res
        e_x = J_x.T @ res
        delta_c = np.linalg.inv(A - B@D_inv@B.T) @ (e_c - B@D_inv@e_x)
        delta_x = D_inv @ (e_x - B.T@delta_c)

        params[0:7*nCams] += delta_c.flatten()
        params[7*nCams:] += delta_x.flatten()

        # Normalize quaternions
        for i in range(nCams):
            params[i*7:i*7+4] = params[i*7:i*7+4]/np.linalg.norm(params[i*7:i*7+4])

        
    return params, res_list


def TestJacobian(cam_params, point_w, cam):
    # point_w = np.concatenate((point_w, [1]))
    f = cam[0]
    K = np.array([[f, 0, 0], [0, f, 0]])
    q = cam_params[0:4]
    if q[0] < 0:
        q = -q
    q = q/np.linalg.norm(q)
    qw, qx, qy, qz = q.flatten()
    R = quat2dcm(q)
    t = cam_params[4:]
    M = np.block([R, t.reshape((3, 1))])
    # X_prime = M @ point_w.reshape((4, 1))
    X_prime = R @ point_w.reshape(3, 1) + t.reshape(3, 1)
    xp, yp, zp = X_prime.flatten()


    # Derivative of X_prime wrt cam_params
    O = np.zeros((1, 3))
    dXp_dR = np.array([[xp, yp, zp, 0, 0, 0, 0, 0, 0], 
                       [0, 0, 0, xp, yp, zp, 0, 0, 0], 
                       [0, 0, 0, 0, 0, 0, xp, yp, zp]])
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
    dXp_dq = dXp_dR @ dR_dq

    print("First Row: ")
    print(4*qw*xp + 2*qz*yp - 2*qy*zp)

    v = np.array([qx, qy, qz]).reshape(3, 1)
    a = np.array([xp, yp, zp]).reshape(3, 1)
    dXp_dq2 = 2*np.block([qw*a + skew(v)@a, v.T@a*np.eye(3) + v@a.T - a@v.T - qw*skew(a)])

    dXp_dt_numerical = GetJacobian_X_prime_t(cam_params, point_w[0:3])
    dXp_dq_numerical = GetJacobian_X_prime_q(cam_params, point_w[0:3])

    dXp_dX = R
    dXp_dX_numerical = GetJacobian_X_prime_X(cam_params, point_w[0:3])

    dp_dXp = -np.array([[1/zp, 0, -xp/zp**2], [0, 1/zp, -yp/zp**2], [0, 0, 0]])
    dp_dXp_numerical = GetJacobian_pi_h_X_prime(X_prime.flatten())

    J_X = K @ dp_dXp @ R
    J_X_numerical = GetJacobian_full_X(cam_params, point_w[0:3], f)

    J_c1 = K @ dp_dXp @ dXp_dR @ dR_dq
    J_c1_numerical = GetJacobian_full_q(cam_params, point_w[0:3], f)

    J_c2 = K @ dp_dXp @ dXp_dt
    J_c2_numerical = GetJacobian_full_t(cam_params, point_w[0:3], f)
    

    print("Analytical dXp_dt: ")
    print(dXp_dt)
    print("Numerical dXp_dt: ")
    print(dXp_dt_numerical)
    print("Analytical dXp_dq (1): ")
    print(dXp_dq)
    print("Analytical dXp_dq (2): ")
    print(dXp_dq2)
    print("Numerical dXp_dq: ")
    print(dXp_dq_numerical)
    print("Analytical dXp_dX: ")
    print(dXp_dX)
    print("Numerical dXp_dX: ")
    print(dXp_dX_numerical)
    print("Analytical dp_dXp: ")
    print(dp_dXp)
    print("Numerical dp_dXp: ")
    print(dp_dXp_numerical)

    print("Analytical J_X: ")
    print(J_X)
    print("Numerical J_X: ")
    print(J_X_numerical)
    print("Analytical J_c1: ")
    print(J_c1)
    print("Numerical J_c1: ")
    print(J_c1_numerical)
    print("Analytical J_c2: ")
    print(J_c2)
    print("Numerical J_c2: ")
    print(J_c2_numerical)



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
    params, res = GaussNewton_sparse(ComputeResidual, ComputeJacobian_sparse, params, cam, nPts, pts_obs, damping_factor=10)
    t1 = time.time()

    # nCams = len(cam)
    # TestJacobian(params[7:14], params[nCams*7+3:nCams*7+6], cam[1])

    print(res)

