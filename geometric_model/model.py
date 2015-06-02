import os

__author__ = 'Kirill'

import cv2
import numpy as np
from math import cos, sin, pi, sqrt, acos

VISUAL_MODE=True
NUM_FRAMES = 500

DUMP_FILE="data_temp.txt"
VIDEO_FILE="myVideo_temp.avi"

D = 400
FPS = 10
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
    if VISUAL_MODE:
        for i in xrange(5):
            plane.append((x, y, i / 10., 1))
    else:
        plane.append((x, y, 3, 1))

def square(plane, z, end):
    for x in xrange(-1,2):
        for y in xrange(-1,2):
            plane.append((x / float(end) * (end - z), y / float(end)* (end - z), z, 1))

def fill_square(plane):
    N = 100
    for x in xrange(N):
        for y in xrange(N):
            plane.append(((N / 2 - x) / float(N / 2), (N / 2 - y) / float(N / 2), 0, 1))

def generate_features():
    plane1 = []

    feature(plane1, -1, 1) # 1
    feature(plane1, -1, 2) # 2
    feature(plane1, 2, 1) # 3
    feature(plane1, 2, 2) # 4
    feature(plane1, 0, -10) # 5
    feature(plane1, 5, -8)  # 6
    feature(plane1, 5, -50) # 7
    feature(plane1, 2, -80) # 8
    feature(plane1, 10, -100) # 9
    feature(plane1, -5, -100) # 10
    # feature(plane1, 20, -150) # 11
    # feature(plane1, 1, -4)    # 12
    # feature(plane1, 100, -150) # 14
    # feature(plane1, -100, -200) # 15
    # feature(plane1, -75, -75)   # 16
    # # feature(plane1, -10, -10000) # 17
    # feature(plane1, 10, -100) # 18
    # feature(plane1, 20, -100) # 19
    # feature(plane1, 30, -100) # 20
    # feature(plane1, 3, -100) # 21
    # feature(plane1, -10, -100) # 22


    '''
    for ii in xrange(100):
        square(plane1, 5 / 100. * ii, 5)
    fill_square(plane1)
    '''

    plane1 = np.array(plane1, np.float32)
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

'''
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


def showMap(features):
    x = []
    y = []
    for f in features:
        x.append(f[0])
        y.append(f[1])
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'ro')
    plt.show()



def main():
    global K
    file = open(DUMP_FILE, "w")
    K = get_camera_matrix()
    features = generate_features()
    v = cv2.VideoWriter(VIDEO_FILE, cv2.cv.CV_FOURCC('M', 'S', 'V', 'C'), FPS, (D, D))
    for frame in xrange(NUM_FRAMES):
        tf = frame
        if ((frame + 100) % 200 < 100):
            frame = 100 - (frame % 100)
        else:
            frame = (frame % 100)

        screen = np.zeros((D,D,3),np.uint8)

        offset = -15 + 30. * frame / 100.
        #camera = [offset, frame / 5., 2 * cos(2 * pi / 30. * offset)]
        camera = [sin(2 * pi / 100. * frame) * 0.5, frame / 10. + 7., 1]
        print(tf, frame, camera)
        #camera = [0, frame / 10. + 7., 1]
        eye = np.array(camera, np.float32)
        look_at = np.array([offset, -10000, 1], np.float32)
        up = np.array([0, 0, 1], np.float32)
        l_F = look_at - eye
        l_f = l_F / np.linalg.norm(l_F)
        l_u = up / np.linalg.norm(up)
        l_s = np.cross(l_f, l_u)
        l_u = np.cross(l_s, l_f)

        r_axis = np.cross(eye, look_at)
        r_axis = r_axis / np.linalg.norm(r_axis)

        # R = rotation_matrix([0, 0, 1], pi / 4) * rotation_matrix(r_axis, 3 * pi / 2.)
        angle = float(frame) / 100 * pi * 2
        angle = cos(angle) * (angle / 16. - pi / 2.)
        R = np.matrix([
            [l_s[0], l_s[1], l_s[2]],
            [l_u[0], l_u[1], l_u[2]],
            [l_f[0], l_f[1], l_f[2]]
        ])
        #R = rotation_matrix([1, 0, 0], -pi / 2) * rotation_matrix([0, 1, 0], angle)

        C = np.array([camera]) # Camera position in world coordinates

        t = -1 * R * C.T        # World origin position in camera coordinates
        Rt = np.concatenate((R, t), axis=1)

        for i, x in enumerate(features):
            s = (K * Rt * x)
            if (s[2,0] < 0): continue
            s = s / s[2,0]
            s = np.rint(s)
            if (not (s[0,0] > 0 and s[1,0] > 0 and s[0,0] < D and s[1,0] < D)):
                continue
            screen[D - s[1,0], s[0,0]] = (255, 255, 255)

            file.write("{} {} {} {:.0f} \n".format(tf, i, 0, s[0,0]))
            file.write("{} {} {} {:.0f} \n".format(tf, i, 1, s[1,0]))

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
    file.flush()
    file.close()
    showMap(features)

cv2.waitKey()
cv2.destroyAllWindows()

if __name__ == '__main__':
    main()