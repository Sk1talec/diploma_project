__author__ = 'Kirill'
import numpy as np
from math import *

def q2r(q):
    r = float(q[0])
    x = float(q[1])
    y = float(q[2])
    z = float(q[3])
    return np.matrix([
                         [r * r + x * x - y * y - z * z, 2 * (x * y - r * z), 2 * (z * x + r * y)],
                         [2 * (x * y + r * z), r * r - x * x + y * y - z * z, 2 * (y * z - r * x)],
                         [2 * (z * x - r * y), 2 * (y * z + r * x), r * r - x * x - y * y + z * z]], np.float32)

def v2q(v, eps=0.000001):
    ar = np.array(v)
    theta = np.linalg.norm(ar)
    if theta < eps:
        return [1, 0, 0, 0]
    else:
        ar =  ar / theta
        ar_n = np.linalg.norm(ar[1:])
        return (cos(theta / 2), sin(theta / 2) * ar[0] / ar_n, sin(theta / 2) * ar[1] / ar_n, sin(theta / 2) * ar[2] / ar_n)

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return float(w), float(x), float(y), float(z)

def q_conjugate(q):
    q = normalize(q)
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(q1, v1):
    v1 = normalize(v1)
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = cos(theta)
    x = x * sin(theta)
    y = y * sin(theta)
    z = z * sin(theta)
    return w, x, y, z

def q_to_axisangle(q):
    w, v = q[0], q[1:]
    theta = acos(w) * 2.0
    return normalize(v), theta


def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v


def qconj(q):
    q_out=-q
    q_out[0]=float(q[0])

    return q_out

def normJac(q):
    r=float(q[0])
    x=float(q[1])
    y=float(q[2])
    z=float(q[3])

    print(np.power((r*r+x*x+y*y+z*z), (-3./ 2.)))
    return np.power((r*r+x*x+y*y+z*z), (-3./ 2.)) * np.matrix([
           [x*x+y*y+z*z, -r*x, -r*y, -r*z],
           [-x*r, r*r+y*y+z*z, -x*y, -x*z],
           [-y*r, -y*x, r*r+x*x+z*z, -y*z],
           [-z*r, -z*x, -z*y, r*r+x*x+y*y]],
                                                   np.float32)
