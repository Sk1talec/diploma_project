import os

__author__ = 'Kirill'

import cv2
import numpy as np
from math import cos, sin, pi, sqrt, acos
from numpy.linalg import norm

D = 400
# x + 10 = 0
Speed = 10
K = 10
N = 1000
'''
for i in xrange(N):
    x = random.random()*K * 2
    y = 0
    z = random.random()*K
    r = (x, y, z, 1)
    plane1.append(r)
'''
'''
plane2 = []
for ii in xrange(N):
    x = random.random()*K * 2
    y = -30
    z = random.random()*K
    r = (x, y, z, 1)
    plane2.append(r)

plane2 = np.array(plane2)
plane2 = plane2[..., np.newaxis]
'''
def feature(plane, x, y):
    plane.append((x, y, 0, 1))
    plane.append((x, y, 5, 1))
    plane.append((x, y, 7, 1))

def generate_features():
    plane1 = []
    feature(plane1, 1, 1)
    feature(plane1, 1, 2)
    feature(plane1, 2, 1)
    feature(plane1, 2, 2)
    plane1 = np.array(plane1)
    return plane1[..., np.newaxis]

def get_camera_matrix():
    f = D
    c_x = D / 2
    c_y = D / 2
    return np.matrix([
        [f, 0, c_x],
        [0, f, c_y],
        [0, 0,  1]
    ], np.float32)

def rotate_camera(x, y, z):
    return get_rotation_matrix(x, y, z)

def get_rotation_matrix(x, y, z):
    X = np.matrix([
        [1,   0   ,    0   ],
        [0, cos(x), -sin(x)],
        [0, sin(x), cos(x) ]
    ], np.float32)
    Y = np.matrix([
        [ cos(y),   0   , sin(y)],
        [   0   ,   1   ,   0   ],
        [-sin(y),   0   , cos(y)]
    ], np.float32)
    Z = np.matrix([
        [ cos(z), -sin(z), 0 ],
        [ sin(z),  cos(z), 0 ],
        [   0   ,    0   , 1 ]
    ], np.float32)
    return X*Y*Z

'''
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/sqrt(np.dot(axis, axis))
    a = cos(theta/2)
    b, c, d = -axis*sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.matrix([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
'''
def main():
    global K
    K = get_camera_matrix()
    features = generate_features()
    v = cv2.VideoWriter("myVideo_temp.avi", cv2.cv.CV_FOURCC('M', 'S', 'V', 'C'), 10, (D, D))
    for frame in xrange(100):
        screen = np.zeros((D,D,3),np.uint8)

        angle = float(frame) / 100 * pi * 2
        R = rotate_camera(0, pi / 2, pi / 2)

        offset = -15 + 30. * frame / 100.
        C = np.array([[0, 0, offset]]) # Camera position in world coordinates

        t = -1 * R.T * C.T        # World origin position in camera coordinates
        Rt = np.concatenate((R.T, t), axis=1)

        for x in features:
            s = (K * Rt * x).round()
            if (s[2,0] < 0): continue
            s = s / s[2,0]
            if (not (s[0,0] > 0 and s[1,0] > 0 and s[0,0] < D and s[1,0] < D)):
                continue
            s = np.array(s, np.uint32)
            screen[D - s[1,0], s[0,0]] = (255, 0, 0)

        '''
        for x in plane2:
            s = np.array(K * Rt * x)
            if (s[2] < 0): continue
            s = s / s[2]
            # print("R", s)
            if (not ((s >= 0).all() and (s < D).all())):
                continue
            s = np.array(s, np.uint32)
            screen[s[0], s[1]] = (0,0,255)
        '''

        v.write(screen)
        print(frame)

cv2.waitKey()
cv2.destroyAllWindows()

if __name__ == '__main__':
    main()