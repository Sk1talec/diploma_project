import cv2
import numpy as np
from math import *

import slam_derivatives
import slam_feature
import quaternions

TRACKS_FILE='data_temp.txt'

FPS = 10
D = 400
f = D
c_x = D / 2
c_y = D / 2
K = np.matrix([
    [f, 0, c_x],
    [0, f, c_y],
    [0, 0, 1]
])

STD_PXL = 1

eps = 1e-10
# Prediction error
error_prediction = 0.00007
# Measurement error
error_measurement = 0.0007
sigma_a = 0.007
# Measurement error
sigma_alpha = 0.0007
sigma_image_noise = 1.0

initial_depth = 1

r_k_k = (0, 0, 0)

N_cam_params = 13  # 3(3d pose) + 4( rotation quaternion ) + 3 ( linear velocity) + 3 ( angular velocity)

FEATURES_DATA = {}

x = np.zeros((N_cam_params, 1), np.float32)  # State vector
#base camera position
x[0] = 0
x[1] = 0
x[2] = 0

#base camera rotation
x[3] = 1
x[4] = 0
x[5] = 0
x[6] = 0

#linear velocity
init_lv = 0.025
x[7] = 1e-15
x[8] = 1e-15
x[9] = 1e-15

#Angular velocity
init_av = 0.025
x[10] = 1e-15
x[11] = 1e-15
x[12] = 1e-15

# State covariance
P = np.matrix(np.diag((eps, eps, eps,
             eps, eps, eps, eps,
            init_lv ** 2, init_lv ** 2, init_lv ** 2,
            init_av ** 2, init_av ** 2, init_av ** 2)))

u = np.zeros((6, 1), np.float32)  # Control vector

z = np.matrix((10, 10))  # Measurement

H = np.matrix((10, 10))  # Measurement Jacobian


# These are given
Q = np.eye(N_cam_params, dtype=np.float32) * error_prediction  # Prediction error
R = np.eye(N_cam_params, dtype=np.float32) * error_measurement  # Measurement error
I = np.identity(10)



# state transition
def f(x, u, dt):
    r_wc = x[:3] + (x[7:10] + u[0:3]) * dt # Camera position. Linear velocity
    a_vel = ((x[10] + u[3]) * dt, (x[11] + u[4]) * dt, (x[12] + u[5]) * dt)
    q_wc = quaternions.q_mult((x[3], x[4], x[5], x[6]), quaternions.v2q(a_vel))
    matr = np.matrix([[r_wc[0], r_wc[1], r_wc[2], q_wc[0], q_wc[1], q_wc[2], q_wc[3]]], np.float32).T
    out = np.concatenate((matr, x[7:]), axis=0)
    return out



#Observation
def h(x):
    pass

def read_frames():
    frames = []
    f_data = open(TRACKS_FILE)
    for i, l in enumerate(f_data.readlines()):
        l = l[:-1]
        if (l[-1] == ' '):
            l = l[:-1]
        l = l.split(' ')
        l[0] = int(l[0])
        l[1] = int(l[1])
        l[2] = int(l[2])
        l[3] = float(l[3])
        if len(frames) <= l[0]:
            frames.append([])
        if i % 2 == 0:
            f_list = [l[1], 0, 0]  #Feature number, feature coordinate
        index = int(l[2]) + 1
        if (TRACKS_FILE=='traces.txt'):
            index-=1
        f_list[index] = l[3]
        if i % 2 == 1:
            frames[l[0]].append(f_list)
    return frames

def predict_camera_measurement(x, frame_features):
    camera_position = x[0:3]
    r_wc = quaternions.q2r(x[3:7])

    for feature in frame_features:
        feature_number = feature[0]
        if (not visible_feature(feature_number)):
            continue
        i = FEATURES_DATA[feature_number].index
        y = x[i : i + 6]
        print(len(x), i, feature_number)
        h = h_inverse_depth(y, camera_position, r_wc)
        if (len(h) > 0):
            FEATURES_DATA[feature_number].h = h

def m(a,b):
    '''
    Unit vector from azimut-elevation angles
    '''
    theta = a
    phi = b
    cphi = cos(phi)
    return np.matrix([cphi *sin(theta),   -sin(phi),  cphi*cos(theta)], np.float32).T

def h_inverse_depth( yinit, camera_pos, r_wc):
    print(yinit)
    r_cw = r_wc.T
    feature_pos = yinit[0:3]
    theta = float(yinit[3])
    phi = float(yinit[4])
    inverse_depth = float(yinit[5])

    unit_feature_direction = m(theta, phi)

    hrl = r_cw * ((feature_pos - camera_pos)*inverse_depth + unit_feature_direction )
    x = float(hrl[0])
    y = float(hrl[1])
    z = float(hrl[2])

    if ((atan2(x, z) * 180/pi < -60) or (atan2(x, z) * 180/pi > 60) or
        (atan2(y, z) * 180/pi < -60) or (atan2(y, z) * 180/pi > 60)):
        return []
    u, v = hu(hrl)

    if ( u > 0 ) and ( u < D) and ( v > 0 ) and ( v < D):
        return np.matrix([[u, v]], np.float32).T
    else:
        return []

def hu(Y):
    x = float(Y[0])
    y = float(Y[1])
    z = float(Y[2])

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

    r_W = x[0:3]
    q_WR = x[3:7]

    h_LR_x=-(U0-u)/float(fku)
    h_LR_y=-(V0-v)/float(fkv)
    h_LR_z=1

    h_LR=np.matrix((h_LR_x, h_LR_y, h_LR_z), np.float32).T

    n=quaternions.q2r(q_WR)*h_LR
    nx=n[0]
    ny=n[1]
    nz=n[2]

    return np.matrix(
        [r_W[0], r_W[1], r_W[2], atan2(nx,nz), atan2(-ny,sqrt(nx*nx+nz*nz)), init_rh], np.float32).T

def init_features(x, P, frame_features):
    global FEATURES_DATA
    ## Add new features
    for feature in frame_features:
        f_number = feature[0]
        if (f_number in FEATURES_DATA and not FEATURES_DATA[f_number].retrack):
            continue

        #TODO support retrack data

        u = feature[1]
        v = feature[2]

        ft = new_feature(x, u, v, initial_depth)
        x=np.concatenate((x, ft), axis=0)
        P = slam_derivatives.add_a_feature_covariance_inverse_depth(P, u, v, x, STD_PXL, 1, K)
        f_data = slam_feature.Feature()
        f_data.uv_init = np.matrix([[u, v]], np.float32)
        f_data.y = ft
        f_data.number=f_number
        f_data.index=len(x) - 6
        FEATURES_DATA[f_number] = f_data
    return x, P


def predict_step(x, P, u, dt):

    # Predict
    x_tmp = f(x, u, dt)  #Predicted state estimate
    F = slam_derivatives.compute_F_cam(x, dt) # State transition Jacobian. Can be expressed as f derivative

    #Linear and angular acceleration noise covariance
    la_noise_cov = (sigma_a * dt) ** 2
    aa_noise_cov = (sigma_alpha * dt) ** 2
    #Q = np.zeros((13, 13), np.float32)
    #Q[:6, :6] = np.diag((la_noise_cov, la_noise_cov, la_noise_cov, aa_noise_cov, aa_noise_cov, aa_noise_cov))


    P_tmp = np.concatenate([
        np.concatenate([F*P[:13,:13]*F.I + Q, F*P[:13, 13:]], axis=1),
        np.concatenate([P[13:,:13]*F.I, P[13:,13:]], axis=1),
                      ], axis=0)

    return x_tmp, P_tmp


def prepare_features():
    global FEATURES_DATA
    for key in FEATURES_DATA:
        FEATURES_DATA[key].visible=False

def visible_feature(f_number):
    return (f_number in FEATURES_DATA) and FEATURES_DATA[f_number].visible


def number_visible_features():
    i = 0
    for feature in FEATURES_DATA:
        if FEATURES_DATA[feature].visible:
            i+=1
    return i


def compute_features_derivatives(x, K, frame_features):
    global FEATURES_DATA
    x_camera = x[:13]
    for feature in (frame_features):
        f_number = feature[0]
        if (not visible_feature(f_number)):
            continue
        i = FEATURES_DATA[f_number].index
        y = x[i : i + 6]
        FEATURES_DATA[f_number].H = slam_derivatives.calculate_H_inverse_depth(x_camera, y, K, FEATURES_DATA[f_number].h, number_visible_features(), i)

def update_features(x, P, frame_features):
    global FEATURES_DATA
    predict_camera_measurement(x, frame_features)
    compute_features_derivatives(x, K, frame_features)

    for feature in (frame_features):
        feature_number = feature[0]
        if (not visible_feature(feature_number)):
            continue
        if len(FEATURES_DATA[feature_number].h) > 0:
            f_data = FEATURES_DATA[feature_number]
            f_data.S = f_data.H * P * f_data.H.T + f_data.R
            f_data.z = np.matrix([[feature[1], feature[2]]], np.float32).T


def update(x_tmp, P_tmp, z, h, H, R):
    if len(z)>0:
        S = H * P_tmp * H.T + R
        K = P_tmp * H.T * S.I

        x = x_tmp + K*(z - h)
        P = P_tmp - K*S*K.T
        P = 0.5*P + 0.5*P.T

        Jnorm = quaternions.normJac(x[3:7])
        P = np.concatenate([
                               np.concatenate([P[0:3,0:3], P[0:3,3:7]*Jnorm.T, P[0:3,7:]], axis=1),
                               np.concatenate([Jnorm*P[3:7,0:3], Jnorm*P[3:7,3:7]*Jnorm.T, Jnorm*P[3:7,7:]], axis=1),
                               np.concatenate([P[7:,0:3], P[7:, 3:7] * Jnorm.T, P[7:,7:]], axis=1)],
                           axis=0)

        x[3:7] = x[3:7] / np.linalg.norm( x[3:7])
        return x, P
    else:
        return x_tmp, P_tmp

def update_step(x_tmp, P_tmp, frame_features):
    global FEATURES_DATA
    #TODO rewrite to speedup!
    z = []
    h = []
    H = []

    for feature in (frame_features[0:]):
        f_number = feature[0]
        if (not visible_feature(f_number) or (len(FEATURES_DATA[f_number].h) == 0)):
            continue
        if (z == []):
            z = FEATURES_DATA[f_number].z
            h = FEATURES_DATA[f_number].h
            H = FEATURES_DATA[f_number].H

        f_data = FEATURES_DATA[f_number]
        z = np.concatenate([z, f_data.z], axis=0)
        h = np.concatenate([h, f_data.h], axis=0)
        H = np.concatenate([H, f_data.H], axis=0)

    R = np.eye(len(z), dtype=np.float32)

    return update(x_tmp, P_tmp, z, h, H, R)


def update_features_indexes(i):
    for number in FEATURES_DATA:
        index = FEATURES_DATA[number].index
        if i < index:
            FEATURES_DATA[number].index -=6
            print("UP_INDEX", number, FEATURES_DATA[number].index, i)



def cut_x_and_p(current_frame, prev_frame):
    global x, P
    found_features = set()
    for p_feature in prev_frame:
        prev_f_num = p_feature[0]
        found=False
        for c_feature in current_frame:
            if c_feature[0] == prev_f_num:
                found=True
                break
        if found:
            #We know the feature and continue tracking it
            #Mark it visible
            if (prev_f_num in FEATURES_DATA):
                print("VISIBLE", prev_f_num)
                FEATURES_DATA[prev_f_num].visible=True
                found_features.add(prev_f_num)

        #Delete feature as we couldn't track it
        if (visible_feature(prev_f_num)):
            #It's known feature disappeared
            i = FEATURES_DATA[prev_f_num].index
            FEATURES_DATA[prev_f_num].visible=False
            print("DEL_F",prev_f_num,i)
            update_features_indexes(i)
            x = np.delete(x, np.s_[i:i + 6], axis=0)
            P = np.delete(P, np.s_[i:i + 6], axis=0)
            P = np.delete(P, np.s_[i:i + 6], axis=1)
            continue

    #Try retrack
    for c_feature in current_frame:
        cur_f_num = c_feature[0]
        if cur_f_num in found_features:
            continue

        if (cur_f_num in FEATURES_DATA and not FEATURES_DATA[cur_f_num].visible):
            #We know that feature but lost track in the past
            print("RETRACK", cur_f_num)
            FEATURES_DATA[cur_f_num].retrack=True

    return x, P


def feature_2_cartesian(id):
    rw = id[0:3]
    theta = float(id[3])
    phi = float(id[4])
    rho = float(id[5])

    cphi = cos(phi)
    m = np.matrix([[cphi * sin(theta), -sin(phi), cphi*cos(theta)]], np.float32).T
    return float(rw[0] + (1./rho) * m[0]), float(rw[1] + (1./rho) * m[1]), float(rw[2] + (1./rho) * m[2])


def print_result(x, frame_features):
    for feature in (frame_features):
        if (not visible_feature(feature[0])):
            continue
        i = FEATURES_DATA[feature[0]].index
        a, b, c = feature_2_cartesian(x[i:i+6])
        print("FEATURE", feature[0], "XYZ cords: ", a, b, c)


def show_plot(trajectory):
    from matplotlib import pyplot
    import pylab
    from mpl_toolkits.mplot3d import Axes3D

    fig = pylab.figure()
    ax = Axes3D(fig)

    x = [trajectory[i][0] for i in xrange(len(trajectory))]
    y = [trajectory[i][1] for i in xrange(len(trajectory))]
    z = [trajectory[i][2] for i in xrange(len(trajectory))]

    ax.scatter(x, y, z)
    pyplot.show()


def update_features_info():
    for key in FEATURES_DATA:
        feature = FEATURES_DATA[key];
        feature.h=[]
        feature.H=0
        feature.z=0
        feature.S=0


def main():

    import sys
    #sys.stdout = open('slam_out.txt', 'w')
    np.set_printoptions(precision=4, suppress=True)

    frames = read_frames()
    global F, H, P, x, u
    dt = 1. / FPS

    cam_trajectory=[]
    for k in xrange(1, len(frames)):
        #TODO Implement dynamic features addition and deletion
        print " "
        print frames[k]
        print("Frame: ", str(k), "Num features:", len(frames[k]))
        print("POSITION", float(x[0]), float(x[1]), float(x[2]))
        #prepare_features()
        x, P = cut_x_and_p(frames[k], frames[k-1])
        update_features_info()
        x, P = init_features(x, P, frames[k - 1]) #From prev frame
        x_tmp, P_tmp = predict_step(x, P, u, dt) # PREDICT STEP

        frame_features = frames[k] # Read new frame

        update_features(x, P, frame_features)
        x, P = update_step(x_tmp, P_tmp, frame_features) # UPDATE STEP

        '''
        f_number=7
        if (f_number in FEATURES_DATA and FEATURES_DATA[f_number].visible):
            i = FEATURES_DATA[f_number].index
            y = x[i:i+6]
            a, b, c = id_2_cart(y)
            cam_trajectory.append([a, b, c])
        '''
        cam_trajectory.append([x[0], x[1], x[2]])
        print_result(x, frame_features)

    cam_trajectory = []
    for key in FEATURES_DATA:
        cam_trajectory.append(feature_2_cartesian(FEATURES_DATA[key].y))
    print("LEN", len(cam_trajectory))
    show_plot(cam_trajectory)

if __name__ == '__main__':
    main()