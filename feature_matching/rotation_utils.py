import numpy as np

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

    # Extract the values from Q
    # q0 = q[0]
    # q1 = q[1]
    # q2 = q[2]
    # q3 = q[3]
    #
    # # First row of the rotation matrix
    # r00 = 2 * (q0 * q0 + q1 * q1) - 1
    # r01 = 2 * (q1 * q2 - q0 * q3)
    # r02 = 2 * (q1 * q3 + q0 * q2)
    #
    # # Second row of the rotation matrix
    # r10 = 2 * (q1 * q2 + q0 * q3)
    # r11 = 2 * (q0 * q0 + q2 * q2) - 1
    # r12 = 2 * (q2 * q3 - q0 * q1)
    #
    # # Third row of the rotation matrix
    # r20 = 2 * (q1 * q3 - q0 * q2)
    # r21 = 2 * (q2 * q3 + q0 * q1)
    # r22 = 2 * (q0 * q0 + q3 * q3) - 1
    #
    # # 3x3 rotation matrix
    # rot_matrix = np.array([[r00, r01, r02],
    #                        [r10, r11, r12],
    #                        [r20, r21, r22]])
    #
    # return rot_matrix

