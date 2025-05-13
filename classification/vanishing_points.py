'''
Author: Dylan Blake
'''
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os
import sys
import matplotlib.pyplot as plt
import heapq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_utils import equirectangular_to_perspective
from homography import homography_RANSAC

def detect_lines(im, visualize=False):
    '''
    run Canny followed by probabilistic Hough Lines
    '''
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    blurred = cv2.GaussianBlur(gray, (5,5), 2.0)
    #cv2.imshow("blurred", blurred)
    edges = cv2.Canny(blurred, 20, 30)
    #cv2.imshow("edges", edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)

    if visualize:    
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(im, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imshow("lines", im)
    return edges, lines

def intersections_and_separation_scores(lines):
    '''
    computes the intersections of a set of lines and the separation score between intersecting lines
    we define the separation score as y_sep / x_sep.
    For a large y_sep and small x_sep, the score is large. This is a good vanishing point candidate.
    For a small y_sep and a large x_sep, the score is small. This is a bad vanishing point candidate.
    '''
    intersections = []
    separation_scores = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            l1 = lines[i][0]
            l2 = lines[j][0]
            pt = intersection(l1, l2)
            y_sep = max_y_separation(l1, l2)
            x_sep = min_x_separation(l1, l2)
            line_similarity = similarity(l1, l2) 
            m_similarity = mirror_similarity(l1, l2)
            if pt is not None:
                intersections.append(pt)
                pt_dst = min_intersection_distance(pt, l1, l2)
                #promote y separation, promote intersection distance, discourage x separation, discourage line similarity, promote mirrored line similarity 
                separation_scores.append(((y_sep + pt_dst) / (x_sep + 1e-3)) * (1-line_similarity) * m_similarity)
    return (np.array(intersections), np.array(separation_scores))
def min_intersection_distance(pt, l1, l2):
    '''
    calculates squared distance to one of l1's endpoints
    '''
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    p1, p2 = pt[:2]

    v1 = np.array([x1-p1, y1-p2], dtype=np.float64)
    v2 = np.array([x2-p1, y2-p2], dtype=np.float64)
    v3 = np.array([x3-p1, y3-p2], dtype=np.float64)
    v4 = np.array([x4-p1, y4-p2], dtype=np.float64)
    return min(np.dot(v1, v1), np.dot(v2, v2), np.dot(v3, v3), np.dot(v4, v4))

def max_y_separation(l1, l2):
    '''
    take the maximum y_separation between endpoint y's in the two lines
    '''
    _, y1, _, y2 = l1
    _, y3, _, y4 = l2
    return max(abs(y3 - y1), abs(y4 - y2))

def min_x_separation(l1, l2):
    '''
    take the minimum x_separation between endpoints only (x4 - x2) (x3 - x1)
    '''
    x1, _, x2, _ = l1
    x3, _, x4, _ = l2
    return min(abs(x3 - x1), abs(x4 - x2))

def intersection(l1, l2):
    '''
    intersection point between two lines
    '''
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    A = np.array([
        [x2 - x1, -(x4 - x3)],
        [y2 - y1, -(y4 - y3)]
    ])
    b = np.array([x3 - x1, y3 - y1])

    if np.linalg.matrix_rank(A) < 2:
        return None  # parallel

    t, _ = np.linalg.solve(A, b)
    intersection = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    return intersection
def similarity(l1, l2):
    '''
    convert lines to vectors, normalize, then dot
    '''
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    v1 = np.array([x2-x1, y2-y1], dtype=np.float64)
    v1 = v1 / np.linalg.norm(v1) 
    v2 = np.array([x4-x3, y4-y3], dtype=np.float64)
    v2 = v2 / np.linalg.norm(v2)
    return np.dot(v1,v2)
def mirror_similarity(l1, l2):
    '''
    mirror l1 across the axis it is aligned with
    convert lines to vectors, normalize, then dot
    '''
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    v1 = np.array([x2-x1, y2-y1], dtype=np.float64)
    v1 = v1 / np.linalg.norm(v1) 
    v2 = np.array([x4-x3, y4-y3], dtype=np.float64)
    v2 = v2 / np.linalg.norm(v2)

    x_axis = (np.dot(v1, np.array([1,0], dtype=np.float64)) + 1.0) / 2.0 #convert from [-1.0, 1.0] to [0.0 - 1.0] domain
    y_axis = (np.dot(v1, np.array([0,1], dtype=np.float64)) + 1.0) / 2.0 #convert from [-1.0, 1.0] to [0.0 - 1.0] domain

    if x_axis > y_axis:
        v1[1] *= -1 #mirror the y component across the x axis 
    else:
        v1[0] *= -1 #mirror the x component across the y axis

    return np.dot(v1,v2)
def cluster_vanishing_points(points, separation_scores):
    '''
    take all intersection points, cluster them, then the vanishing point is the mean of the points in the cluster.
    We also rank these vanishing points by the cluster's separation score.
    This score encourages vanishing points resulting from unique features.
    '''
    clustering = DBSCAN(eps=20, min_samples=2).fit(points) #DBSCAN(eps=60, min_samples=5).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    vp_pq = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label]
        cluster_separations = separation_scores[labels == label]
        vp = np.mean(cluster_points, axis=0)
        sc = np.max(cluster_separations)
        heapq.heappush(vp_pq, (-sc, vp))

    vanishing_points = []
    while len(vp_pq) > 0:
        _, vp = heapq.heappop(vp_pq)
        vanishing_points.append(vp) 
        if len(vanishing_points) > 5: #we only search for two points: vanishing point left and vanishing point right 
            #if vanishing_points[0][0] > vanishing_points[1][0]: #ensure vanishing point order: left comes before right.
            #    tmp = vanishing_points[0][0]
            #    vanishing_points[0][0] = vanishing_points[1][0]
            #    vanishing_points[1][0] = tmp
            break
    return np.array(vanishing_points)

def get_vanishing_points(lines, visualize=False, im=None):
    '''
    input: 
    lines: obtained from detect_lines
    im_w: perspective image width for contrived vertical vanishing point
    output:
    vanishing point left, vanishing point right
    '''
    #for line in lines:
    #    x1, y1, x2, y2 = line[0]
    #    cv2.line(im, (x1,y1), (x2,y2), (0,255,0), 2)
    #cv2.imshow("lines", im)
    intersections, separation_scores = intersections_and_separation_scores(lines)
    vanishing_points = cluster_vanishing_points(intersections, separation_scores) #only obtain left and right vanishing points
    if visualize:
        plt.imshow(im)
        plt.scatter(intersections[:,0], intersections[:,1], label='Line Intersections', s=10, c='blue', marker='o')
        plt.scatter(vanishing_points[:2,0], vanishing_points[:2,1], label='Vanishing Points', s=10, c='red', marker='o')
        plt.legend()
        plt.show()
    #assert vanishing_points.shape[0] == 2, "cannot proceed without vl and vr"
    vanishing_points = np.hstack((vanishing_points[:2], np.ones(shape=(2, 1)))) #ones column
    vl = vanishing_points[0]
    vr = vanishing_points[1]
    return (vl, vr)

if __name__ == '__main__':
    directory = 'C:\\UofM CompSci Masters\\Computer Vision 5561\\Project\\Dataset\\data\\0016\\panos'
    filename = 'floor_02_partial_room_04_pano_17.jpg'
    full_path = os.path.join(directory, filename)
    im = cv2.imread(full_path, flags=cv2.IMREAD_COLOR) #Same as IMREAD_COLOR_BGR

    visualize=False
    #cv2.imshow("original", im)

    im = equirectangular_to_perspective(im, 120, 0, 0, 960, 480)
    #cv2.imshow("perspective", im)
    im_canny, lines = detect_lines(im, visualize)
    vl, vr, vv = get_vanishing_points(lines, im.shape[1], visualize)
    H = homography_RANSAC()    

    
    #output_directory = os.path.join('./', filename[:-4])
    #if not os.path.exists(output_directory):
    #    os.makedirs(output_directory)
    #plt.imsave(os.path.join('./', filename[:-4], 'vanishing_points.png'), cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    