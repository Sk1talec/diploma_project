import cv2
import numpy as np
from math import *

import slam_derivatives
import slam_feature
import quaternions

FPS = 10
D = 400
f = D
c_x = D / 2
c_y = D / 2
K = np.array([
    [f, 0, c_x],
    [0, f, c_y],
    [0, 0, 1]
])

STD_PXL = 1

eps = 1e-10
sigma_a = 0.007
sigma_alpha = 0.007
sigma_image_noise = 1.0

initial_depth = 1

r_k_k = (0, 0, 0)

N_features = 16
N_cam_params = 13  # 3(3d pose) + 4( rotation quaternion ) + 3 ( linear velocity) + 3 ( angular velocity)
N = N_cam_params + N_features * 6

FEATURES_INITIALISED = [False for i in xrange(N_cam_params)]

x = np.zeros((1, N), np.float32)  # State vector
#base camera position
x[0, 0] = -15
x[0, 1] = 8
x[0, 2] = 1

#base camera rotation
x[0, 3] = 1
x[0, 4] = 0
x[0, 5] = 0
x[0, 6] = 0

#linear velocity
init_lv = 0.025
x[0, 7] = init_lv
x[0, 8] = init_lv
x[0, 9] = init_lv

#Angular velocity
init_av = 0.025
x[0, 10] = init_av
x[0, 11] = init_av
x[0, 12] = init_av

# State covariance
P = np.matrix(np.diag((eps, eps, eps,
             eps, eps, eps, eps,
            init_lv ** 2, init_lv ** 2, init_lv ** 2,
            init_av ** 2, init_av ** 2, init_av ** 2)))

u = np.zeros((1, 6), np.float32)  # Control vector

z = np.matrix((10, 10))  # Measurement

H = np.matrix((10, 10))  # Measurement Jacobian


# These are given
Q = np.eye(N_cam_params, dtype=np.float32) * sigma_alpha  # Prediction error
R = np.eye(N_cam_params, dtype=np.float32) * sigma_alpha  # Measurement error
I = np.identity(10)



# state transition
def f(x, u, dt):
    print(x[0,7:10] , u[0,0:3])
    r_wc = x[0,:3] + (x[0,7:10] + u[0,0:3]) * dt # Camera position. Linear velocity

    a_vel = ((x[0, 10] + u[0, 3]) * dt, (x[0, 11] + u[0, 4]) * dt, (x[0, 12] + u[0, 5]) * dt)
    q_wc = quaternions.q_mult((x[0, 3], x[0, 4], x[0, 5], x[0, 6]), quaternions.v2q(a_vel))

    matr = np.matrix([[r_wc[0], r_wc[1], r_wc[2], q_wc[0], q_wc[1], q_wc[2], q_wc[3]]], np.float32)
    x_arr = x[0, 7:][np.newaxis]
    out = np.concatenate((matr, x_arr), axis=1)
    return out



#Observation
def h(x):
    pass

def read_frames():
    frames = []
    f_data = open('data.txt')
    for i, l in enumerate(f_data.readlines()):
        l = map(int, l.split(' ')[:-1])
        if len(frames) <= l[0]:
            frames.append([])
        if i % 2 == 0:
            f_list = [l[1], 0, 0]  #Feature number, feature coordinate
        f_list[int(l[2]) + 1] = l[3]
        if i % 2 == 1:
            frames[l[0]].append(f_list)
    return frames

def predict_camera_measurement(x, P, frame_features):
    out_h = []

    camera_position = x[0, 0:3]
    r_wc = quaternions.q2r(x[0, 3:7])

    for feature in frame_features:
        feature_number = feature[0]
        y = x[0, N_cam_params + 6 * feature_number : N_cam_params + 6 * (feature_number + 1)]
        h = h_inverse_depth(y, camera_position, r_wc)
        if (h != []):
            out_h.append((feature_number, np.matrix([
                [h[0]],
                [h[1]]
            ], np.float32)))

def m(a,b):
    '''
    Unit vector from azimut-elevation angles
    '''
    theta = a
    phi = b
    cphi = cos(phi)
    return np.matrix([cphi *sin(theta),   -sin(phi),  cphi*cos(theta)], np.float32).T

def h_inverse_depth( yinit, camera_pos, r_wc):
    r_cw = r_wc.T

    feature_pos = yinit[0:3]
    theta = yinit[3]
    phi = yinit[4]
    inverse_depth = yinit[5]

    unit_feature_direction = m(theta, phi)

    hrl = r_cw * ((feature_pos - camera_pos)*inverse_depth + unit_feature_direction )
    x = hrl[0, 0]
    y = hrl[0, 1]
    z = hrl[0, 2]

    if ((atan2(x, z) * 180/pi < -60) or (atan2(x, z) * 180/pi > 60) or
        (atan2(y, z) * 180/pi < -60) or (atan2(y, z) * 180/pi > 60)):
        return []

    u, v = hu(hrl)

    if ( u > 0 ) and ( u < D) and ( v > 0 ) and ( v < D):
        return u, v
    else:
        return []

def hu(Y):
    x = Y[0]
    y = Y[1]
    z = Y[2]

    u0 = K[0,2]
    v0 = K[1,2]
    f  = K[1,1]
    ku = 1 #1/cam.dx;
    kv = 1 #1/cam.dy;

    return u0 + (x / z)*f*ku, v0 + (y / z)*f*kv

def new_feature(x, u, v, init_rh):
    fku = K[0,0]
    fkv = K[1,1]
    U0  = K[0,2]
    V0  = K[1,2]

    r_W = x[0, 0:3]
    q_WR = x[0, 3:7]

    h_LR_x=-(U0-u)/float(fku)
    h_LR_y=-(V0-v)/float(fkv)
    h_LR_z=1

    h_LR=np.matrix((h_LR_x, h_LR_y, h_LR_z), np.float32).T

    n=quaternions.q2r(q_WR)*h_LR
    nx=n[0, 0]
    ny=n[0, 1]
    nz=n[0, 2]

    return np.matrix(
        [r_W[0], r_W[1], r_W[2], atan2(nx,nz), atan2(-ny,sqrt(nx*nx+nz*nz)), init_rh], np.float32)

def update_features(x, P, frame_features):
    features_h = predict_camera_measurement(x, P, frame_features)

    ## Add new features
    for feature in frame_features:
        if (FEATURES_INITIALISED[feature[0]]):
            continue
        u = feature[1]
        v = feature[2]
        #TODO Add correct order feature initialisation
        x[0, N_cam_params + 6 * feature[0] : N_cam_params + 6 * (feature[0] + 1)] = new_feature(x, u, v, initial_depth)
        slam_derivatives.add_a_feature_covariance_inverse_depth(P, u, v, x, STD_PXL, 1, K)

        FEATURES_INITIALISED[feature[0]]=True



# P_tmp is P[k, k - 1]
# P is either P[k - 1, k - 1] or P[k, k]

# x_tmp is x[k, k - 1]
# x is either x[k - 1, k - 1] or x[k, k]

def main():

    frames = read_frames()
    global P, x
    dt = 1. / FPS
    for k in xrange(len(frames)):
        frame_featurs = frames[k]

        update_features(x, P, frame_featurs)

        # Predict
        x_tmp = f(x, u, dt)  #Predicted state estimate
        F = slam_derivatives.compute_F(x, dt) # State transition Jacobian. Can be expressed as f derivative
        #Linear and angular acceleration noise covariance
        la_noise_cov = (sigma_a * dt) ** 2
        aa_noise_cov = (sigma_alpha * dt) ** 2
        #Q = np.zeros((13, 13), np.float32)
        #Q[:6, :6] = np.diag((la_noise_cov, la_noise_cov, la_noise_cov, aa_noise_cov, aa_noise_cov, aa_noise_cov))

        P_tmp = F * P * F.T + Q  #Predicted covariance estimate
        print(P_tmp)

        #TODO z, h, H
        #TODO 2. Q, R

        #Update
        y = z - h(x_tmp)  # Innovation or measurement residual
        S = H[k] * P_tmp * H[k].t() + R  # Innovation (or residual) covariance
        K = P_tmp * H[k].t() * S.invert()  # Near-optimal Kalman gain
        x = x_tmp + K * y  # Updated state estimate
        P = (I - K * H[k]) * P_tmp  # Updated covariance estimate


if __name__ == '__main__':
    main()