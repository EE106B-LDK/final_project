#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Utils for C106B grasp planning project.
Author: Chris Correa.
Adapted for Spring 2020 by Amay Saxena
"""
import numpy as np
from trimesh import proximity
from scipy.spatial.transform import Rotation
from casadi import dot
import cv2
from cv2 import approxPolyDP

def find_intersections(mesh, p1, p2):
    """
    Finds the points of intersection between an input mesh and the
    line segment connecting p1 and p2.

    Parameters
    ----------
    mesh (trimesh.base.Trimesh): mesh of the object
    p1 (3x np.ndarray): line segment point
    p2 (3x np.ndarray): line segment point

    Returns
    -------
    on_segment (2x3 np.ndarray): coordinates of the 2 intersection points
    faces (2x np.ndarray): mesh face numbers of the 2 intersection points
    """
    ray_origin = (p1 + p2) / 2
    ray_length = np.linalg.norm(p1 - p2)
    ray_dir = (p2 - p1) / ray_length
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[ray_origin, ray_origin],
        ray_directions=[ray_dir, -ray_dir],
        multiple_hits=True)
    if len(locations) <= 0:
        return [], None
    dist_to_center = np.linalg.norm(locations - ray_origin, axis=1)
    dist_mask = dist_to_center <= (ray_length / 2) # only keep intersections on the segment.
    on_segment = locations[dist_mask]
    faces = index_tri[dist_mask]
    return on_segment, faces

def find_grasp_vertices(mesh, p1, p2):
    """
    If the tips of an ideal two fingered gripper start off at
    p1 and p2 and then close, where will they make contact with the object?
    
    Parameters
    ----------
    mesh (trimesh.base.Trimesh): mesh of the object
    p1 (3x np.ndarray): starting gripper point
    p2 (3x np.ndarray): starting gripper point

    Returns
    -------
    locations (nx3 np.ndarray): coordinates of the closed gripper's n contact points
    face_ind (nx np.ndarray): mesh face numbers of the closed gripper's n contact points
    """
    ray_dir = p2 - p1
    locations, index_ray, face_ind = mesh.ray.intersects_location(
        ray_origins=[p1, p2],
        ray_directions=[p2 - p1, p1 - p2],
        multiple_hits=False)
    return locations, face_ind

def normal_at_point(mesh, p):
    """
    Returns the normal vector to the mesh at a point p.
    Requires that p is a point on the surface of the mesh (or at least
    that it is very close to a point on the surface).
    
    Parameters
    ----------
    mesh (trimesh.base.Trimesh): mesh of the object
    p (3x np.ndarray): point to get normal at

    Returns
    -------
    (3x np.ndarray): surface normal at p
    """
    point, dist, face = proximity.closest_point(mesh, [p])
    if dist > 0.001:
        print("Input point is not on the surface of the mesh!")
        return None
    return mesh.face_normals[face[0]]

def normalize(vec):
    """
    Returns a normalized version of a numpy vector

    Parameters
    ----------
    vec (nx np.ndarray): vector to normalize

    Returns
    -------
    (nx np.ndarray): normalized vector
    """
    return vec / np.linalg.norm(vec)

def length(vec):
    """
    Returns the length of a 1 dimensional numpy vector

    Parameters
    ----------
    vec : nx1 :obj:`numpy.ndarray`

    Returns
    -------
    float
        ||vec||_2^2
    """
    return np.sqrt(vec.dot(vec))

def vec(*args):
    """
    all purpose function to get a numpy array of random things.  you can pass
    in a list, tuple, ROS Point message.  you can also pass in:
    vec(1,2,3,4,5,6) which will return a numpy array of each of the elements 
    passed in: np.array([1,2,3,4,5,6])
    """
    if len(args) == 1:
        if type(args[0]) == tuple:
            return np.array(args[0])
        elif ros_enabled and type(args[0]) == Point:
            return np.array((args[0].x, args[0].y, args[0].z))
        else:
            return np.array(args)
    else:
        return np.array(args)

def hat(v):
    """
    See https://en.wikipedia.org/wiki/Hat_operator or the MLS book

    Parameters
    ----------
    v (3x, 3x1, 6x, or 6x1 np.ndarray): vector to create hat matrix for

    Returns
    -------
    (3x3 or 6x6 np.ndarray): the hat version of the v
    """
    if v.shape == (3, 1) or v.shape == (3,):
        return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
    elif v.shape == (6, 1) or v.shape == (6,):
        return np.array([
                [0, -v[5], v[4], v[0]],
                [v[5], 0, -v[3], v[1]],
                [-v[4], v[3], 0, v[2]],
                [0, 0, 0, 0]
            ])
    else:
        raise ValueError

def adj(g):
    """
    Adjoint of a rotation matrix. See the MLS book.

    Parameters
    ----------
    g (4x4 np.ndarray): homogenous transform matrix

    Returns
    -------
    (6x6 np.ndarray): adjoint matrix
    """
    if g.shape != (4, 4):
        raise ValueError

    R = g[0:3,0:3]
    p = g[0:3,3]
    result = np.zeros((6, 6))
    result[0:3,0:3] = R
    result[0:3,3:6] = np.matmul(hat(p), R)
    result[3:6,3:6] = R
    return result

def look_at_general(origin, direction):
    """
    Creates a homogenous transformation matrix at the origin such that the 
    z axis is the same as the direction specified. There are infinitely 
    many of such matrices, but we choose the one where the y axis is as 
    vertical as possible.  

    Parameters
    ----------
    origin (3x np.ndarray): origin coordinates
    direction (3x np.ndarray): direction vector

    Returns
    -------
    (4x4 np.ndarray): homogenous transform matrix
    """
    up = np.array([0, 0, 1])
    z = normalize(direction) # create a z vector in the given direction
    x = normalize(np.cross(up, z)) # create a x vector perpendicular to z and up
    y = np.cross(z, x) # create a y vector perpendicular to z and x

    result = np.eye(4)

    # set rotation part of matrix
    result[0:3,0] = x
    result[0:3,1] = y
    result[0:3,2] = z

    # set translation part of matrix to origin
    result[0:3,3] = origin

    return result

def look_at_rotated(origin, direction, theta):
    up = np.array([0, 0, 1])
    z = normalize(direction) # create a z vector in the given direction
    x = normalize(np.cross(up, z)) # create a x vector perpendicular to z and up
    y = np.cross(z, x) # create a y vector perpendicular to z and x

    # Rotate around z by theta
    R = Rotation.from_rotvec(theta * z).as_dcm()
    x = np.matmul(R, x)
    y = np.matmul(R, y)

    result = np.eye(4)

    # set rotation part of matrix
    result[0:3,0] = x
    result[0:3,1] = y
    result[0:3,2] = z

    # set translation part of matrix to origin
    result[0:3,3] = origin

    return result

def create_transform_matrix(rotation_matrix, translation_vector):
    """
    Creates a homogenous 4x4 matrix representation of this transform

    Parameters
    ----------
    rotation_matrix (3x3 np.ndarray): Rotation between two frames
    translation_vector (3x np.ndarray): Translation between two frames

    """
    return np.r_[np.c_[rotation_matrix, translation_vector],[[0, 0, 0, 1]]]

def rotation_from_quaternion(q_wxyz):
    """Convert quaternion array to rotation matrix.
    Parameters
    ----------
    q_wxyz : :obj:`numpy.ndarray` of float
        A quaternion in wxyz order.
    Returns
    -------
    :obj:`numpy.ndarray` of float
        A 3x3 rotation matrix made from the quaternion.
    """
    # q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

    r = Rotation.from_quat(q_wxyz)
    try:
        mat = r.as_dcm()
    except:
        mat = r.as_matrix()
    return mat


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True
    """
    try:
        r = Rotation.from_dcm(matrix)
    except:
        r = Rotation.from_matrix(matrix)
    return r.as_quat()

def R_z(theta):
    """Returns 3x3 rotation matrix about z."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def cone_constraints(f, num_facets, mu, gamma):
    """
    Return Casadi constraints for the friction cone.
    Assumes only two contact points.
    """
    # Find facet edges
    v = np.array([mu, 0, 1])
    thetas = np.linspace(0, 2 * np.pi, num=num_facets, endpoint=False)
    vs = [np.matmul(R_z(theta), v) for theta in thetas]
    
    # Find facet normals
    facet_normals = [np.cross(vs[-1], vs[0])]
    for i in range(num_facets - 1):
        v1, v2 = vs[i], vs[i + 1]
        facet_normals.append(normalize(np.cross(v1, v2)))
    return facet_normals

def triangulate(pts0, pts1, K, R, t):
    N = len(pts0)
    pts3d = np.zeros((N, 3))
    P1 = np.matmul(K, np.hstack([np.eye(3), np.zeros((3, 1))]))
    P2 = np.matmul(K, np.hstack([R, t.reshape((3, 1))]))

    # for every point correspondence
    for k in range(pts0.shape[0]):
        # create system of equations that equals to zero vector
        x1, y1 = pts0[k]
        x2, y2 = pts1[k]
        A = np.array([
            P1[0, :] - x1 * P1[2, :],
            P1[1, :] - y1 * P1[2, :],
            P2[0, :] - x2 * P2[2, :],
            P2[1, :] - y2 * P2[2, :]
        ])

        # run SVD on system of equations and get last column of V as solution
        _, _, vh = np.linalg.svd(np.matmul(A.T, A))
        pt3d = vh[-1]

        # get world coordinates of point
        pts3d[k] = (pt3d / pt3d[-1])[:-1]
    return pts3d

COLORS = {
    'blue':    np.array([  255,   0, 0]),
    # 'orange':  np.array([255, 127,   0]),
    # 'yellow':  np.array([255, 255,   0]),
    # 'green':   np.array([  0, 255,   0]),
    # 'purple':  np.array([127,   0, 255]),
    # 'pink':    np.array([255,   0, 255])
}

def detect_face(img, color, delta=np.array([10, 10, 10]), angle_eps=5):
    """
    Returns corners of detected face
    """    
    # Blur and sharpen image and covert from RGB to HSV for easy color detection
    blur = cv2.GaussianBlur(img, (5,5), 0)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, kernel)
    img = cv2.cvtColor(sharpen, cv2.COLOR_RGB2HSV)

    # Mask out colored face and covert to grayscale
    c = COLORS[color].reshape((1, 1, 3)).astype('uint8')
    color_val = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    gray = cv2.inRange(img, color_val - delta, color_val + delta)

    # Perform Canny edge detection and find contours
    edged = cv2.Canny(gray, 30, 200)
    contours = cv2.findContours(edged, cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    # Approximate contours to polygons and return face
    for contour in contours:
        corners = approxPolyDP(contour, cv2.arcLength(contour, True)*.01, True)

        # Check if parallelogram
        if len(corners) == 4:
            def angle(l, c, r): # Angle with center c
                v_0, v_1 = c - l, r - c
                return np.arccos((v_0 * v_1)/(np.linalg.norm(v_0), np.linalg.norm(v_1)))
            angles = []
            for i in range(4):
                l, c, r = corners[(i-1)%4], corners[i], corners[(i+1)%4]
                angles.append(angle(l, c, r))
            if (((angles[0] - angles[1]) < angle_eps) and 
                ((angles[2] - angles[3]) < angle_eps) and
                (np.abs(np.sum(angles) - 180) < angle_eps)):
                return corners
    return None

def detect_face_2(img, color, delta=np.array([64, 200, 200]), th=0.01, vis=False):
    """
    Returns corners of detected face
    """
    # Blur and sharpen image and covert from RGB to HSV for easy color detection
    blur = cv2.GaussianBlur(img, (5,5), 2)
    img = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    if vis:
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Mask out colored face and covert to grayscale
    c = COLORS[color].reshape((1, 1, 3)).astype('uint8')
    color_val = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    gray = cv2.inRange(img, color_val - delta, color_val + delta)
    
    if vis:
        cv2.imshow(color,gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst,None)

    if vis:
        cv2.imshow('harris corners',dst/dst.max())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # All keypoints
    x, y = np.where(dst>th*dst.max())
    pts = np.array([x, y]).T
    if len(pts) < 4:
        return None

    # TODO: Filter keypoints for outliers, or else this won't work well (yikers)

    # Get bbox
    min_pts, max_pts = np.min(pts, axis=0), np.max(pts, axis=0)
    (min_x, min_y), (max_x, max_y) = min_pts, max_pts

    def closest(keypts, pt, other, lam=2.0):
        corner_dists = np.linalg.norm(keypts - pt, axis=1, ord=1)
        other_dists = [np.linalg.norm(keypts - o, axis=1, ord=1) for o in other]
        if len(other_dists) > 0:
            other_dists = np.min(other_dists, axis=0)
            idx = np.argmin(corner_dists - lam * other_dists, axis=0)
        else:
            idx = np.argmin(corner_dists, axis=0)
        return keypts[idx]
    
    bbox_corners = [
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
        [min_x, min_y]
    ]

    # Get corners
    corners = []
    for corner in bbox_corners:
        corners.append(closest(pts, corner, corners))

    if vis:
        print(corners)
        # Threshold for an optimal value, it may vary depending on the image.
        img[:,:] = [0, 0, 0]
        corner_colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255]
        ]
        for i, corner in enumerate(corners):
            x, y = corner
            x, y = int(x), int(y)       
            img[x, y] = corner_colors[i]
        img = cv2.dilate(img,None)
        cv2.imshow('corners',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return corners

def detect_face_3(img, color, delta=np.array([64, 200, 200]), th=0.01, vis=False):
    """
    Returns corners of detected face
    """
    # Blur and sharpen image and covert from RGB to HSV for easy color detection
    blur = cv2.GaussianBlur(img, (5,5), 2)
    img = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    if vis:
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Mask out colored face and covert to grayscale
    c = COLORS[color].reshape((1, 1, 3)).astype('uint8')
    color_val = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    gray = cv2.inRange(img, color_val - delta, color_val + delta)
    
    if vis:
        cv2.imshow(color,gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)
    # dst = cv2.cornerHarris(gray,2,3,0.04)

    # if vis:
    #     cv2.imshow('harris corners',dst/dst.max())
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # All keypoints
    # x, y = np.where(dst>th*dst.max())
    # pts = np.array([x, y]).T
    # if len(pts) < 4:
    #     return None    
    # return pts
    return kp, des

def apply_transform(g, pts):
    """
    pts: Nx3
    returns: Nx3
    """
    N = pts.shape[0]
    pts = np.hstack([pts, np.ones((N, 1))]) # Nx4
    pts = np.matmul(g, pts.T) # 4xN
    pts = (pts / pts[-1])[:-1].T # Nx3
    return pts

if __name__ == '__main__':
    img = cv2.imread('cube.jpg')
    for c in COLORS:
        print(detect_face_2(img, c, vis=True))