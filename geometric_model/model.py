import os

__author__ = 'Kirill'

import cv2
import numpy as np
import random

#Добавить шум к проекциям
#Возвращать 3D массив. Номер кадра, Номер точки, Номер координаты. Хранится координата.
#В результате хотим алгоритм structure motion
'''
Хотим найти свою координату и координаты всех точек.
'''
# x + 10 = 0
Speed = 10
K = 10
N = 1000
plane1 = []
for i in xrange(N):
    x = random.random()*K * 2
    y = 0
    z = random.random()*K
    r = (x, y, z, 1)
    plane1.append(r)

plane1 = np.array(plane1)
plane1 = plane1[..., np.newaxis]

plane2 = []
for ii in xrange(N):
    x = random.random()*K * 2
    y = -30
    z = random.random()*K
    r = (x, y, z, 1)
    plane2.append(r)

plane2 = np.array(plane2)
plane2 = plane2[..., np.newaxis]

v = cv2.VideoWriter("myVideo.avi", cv2.cv.CV_FOURCC('M', 'S', 'V', 'C'), 10, (400, 400))
for frame in xrange(100):
    screen = np.zeros((400,400,3),np.uint8)

    f = 2
    c_x = 200
    c_y = 200
    K = np.array([
        [f, 0, c_x],
        [0, f, c_y],
        [0, 0,  1]
    ])
    K = np.concatenate((K,np.zeros((3,1))), axis=1)
    K = np.asmatrix(K)

    R = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    R = np.concatenate((R,np.zeros((1,3))))
    T = np.array([[-10, 15, -frame / 100. * Speed, 1]])

    P = K * np.concatenate((R, T.T), axis=1)

    for x in plane1:
        s = np.array(P * x)
        if (s[2] < 0): continue
        s = s / s[2]
        # print("B", s)
        if (not ((s >= 0).all() and (s< 400).all())):
            continue
        s = np.array(s, np.uint32)
        screen[s[0], s[1]] = (255,0,0)

    for x in plane2:
        s = np.array(P * x)
        if (s[2] < 0): continue
        s = s / s[2]
        # print("R", s)
        if (not ((s >= 0).all() and (s < 400).all())):
            continue
        s = np.array(s, np.uint32)
        screen[s[0], s[1]] = (0,0,255)

    v.write(screen)
    print(frame)

# cv2.namedWindow("Hi")
# cv2.imshow("Hi", screen)
cv2.waitKey()

cv2.destroyAllWindows()

import matplotlib.pyplot as plot
