__author__ = 'Kirill'

import numpy as np
import quaternions
from math import *


def add_a_feature_covariance_inverse_depth(P, u, v, Xv, std_pxl, std_rho, K):

    fku = K[0, 0]
    fkv = K[1, 1]
    U0 = K[0, 2]
    V0 = K[1, 2]

    q_wc = Xv[3:7]
    R_wc = quaternions.q2r(q_wc)

    uu = u
    vu = v

    X_c = -(U0-uu)/fku
    Y_c = -(V0-vu)/fkv
    Z_c = 1

    XYZ_c = np.matrix([X_c, Y_c, Z_c], np.float32).T

    XYZ_w = R_wc*XYZ_c
    X_w = float(XYZ_w[0])
    Y_w = float(XYZ_w[1])
    Z_w = float(XYZ_w[2])

    dtheta_dgw = np.matrix([Z_w/(X_w ** 2+Z_w ** 2), 0, -X_w/(X_w ** 2+Z_w ** 2)], np.float32)
    dphi_dgw = np.matrix([(X_w * Y_w)/((X_w ** 2 + Y_w ** 2 + Z_w ** 2) * sqrt(X_w ** 2 + Z_w ** 2)),
                 -sqrt(X_w ** 2 + Z_w ** 2)/(X_w ** 2 + Y_w ** 2 + Z_w ** 2),
                 (Z_w * Y_w)/((X_w ** 2 + Y_w ** 2 + Z_w ** 2)*sqrt(X_w ** 2 + Z_w ** 2))], np.float32)
    dgw_dqwr = dRq_times_a_by_dq( q_wc, XYZ_c )

    dtheta_dqwr = dtheta_dgw*dgw_dqwr
    dphi_dqwr = dphi_dgw*dgw_dqwr
    dy_dqwr = np.concatenate((
                                  np.zeros((3,4), np.float32),
                                  dtheta_dqwr,
                                  dphi_dqwr,
                                  np.zeros((1,4), np.float32)),
                             axis=0)

    dy_drw = np.concatenate((
                                np.eye(3, dtype=np.float32),
                                np.zeros((3,3), np.float32)),
                            axis=0)

    dy_dxv = np.concatenate((
                                dy_drw,
                                dy_dqwr,
                                np.zeros((6,6), np.float32)),
                            axis=1)

    dyprima_dgw = np.concatenate((
                                     np.zeros((3,3), np.float32),
                                     dtheta_dgw,
                                     dphi_dgw ),
                                 axis=0)
    dgw_dgc = R_wc
    dgc_dhu = np.matrix([
                            [+1/fku, 0, 0],
                            [0, +1/fkv, 0]
                        ], np.float32).T

    #dhu_dhd = jacob_undistor_fm( cam , uvd );
    #dyprima_dhd = dyprima_dgw*dgw_dgc*dgc_dhu*dhu_dhd;
    dyprima_dhd = dyprima_dgw*dgw_dgc*dgc_dhu

    dy_dhd = np.concatenate((np.concatenate((
                                                dyprima_dhd,
                                                np.zeros((5,1), np.float32)),
                                            axis=1),
                              np.concatenate((
                                                 np.zeros((1,2), np.float32),
                                                 np.eye(1,1,dtype=np.float32)),
                                             axis=1)),
                             axis=0)

    Ri = np.eye(2)*std_pxl ** 2

    Padd = np.concatenate((np.concatenate((
                                              Ri,
                                              np.zeros((2,1), np.float32)),
                                          axis=1),
                           np.concatenate((
                                              np.zeros((1,2)),
                                              np.eye(1,1,dtype=np.float32) * (std_rho ** 2)),
                                          axis=1)),
                          axis=0)

    P_xv = P[0:13, 0:13]
    P_yxv = P[13:, 0:13]
    P_y = P[13:, 13:]
    P_xvy = P[0:13, 13:]
    return np.concatenate((np.concatenate((P_xv, P_xvy, P_xv*dy_dxv.T), axis=1),
                            np.concatenate((P_yxv, P_y, P_yxv*dy_dxv.T), axis=1),
                            np.concatenate((dy_dxv*P_xv, dy_dxv*P_xvy, dy_dxv*P_xv*dy_dxv.T + dy_dhd*Padd*dy_dhd.T), axis=1)),
                           axis=0)

def dRq_times_a_by_dq(q,aMat):

  out=np.zeros((3,4), np.float32)

  TempR = dR_by_dq0(q)
  Temp31 = TempR * aMat
  out[0:3,0]=Temp31.T

  TempR = dR_by_dqx(q)
  Temp31 = TempR * aMat
  out[0:3,1]=Temp31.T

  TempR = dR_by_dqy(q)
  Temp31 = TempR * aMat
  out[0:3,2]=Temp31.T

  TempR = dR_by_dqz(q)
  Temp31 = TempR * aMat
  out[0:3,3]=Temp31.T

  return out


def dR_by_dq0(q):
    q0 = float(q[0])
    qx = float(q[1])
    qy = float(q[2])
    qz = float(q[3])

    return np.matrix([
        [2*q0, -2*qz,  2*qy],
        [2*qz,  2*q0, -2*qx],
        [-2*qy,  2*qx,  2*q0]], np.float32)

def dR_by_dqx(q):
    q0 = float(q[0])
    qx = float(q[1])
    qy = float(q[2])
    qz = float(q[3])

    return np.matrix([
        [2*qx, 2*qy,   2*qz],
        [2*qy, -2*qx, -2*q0],
        [2*qz, 2*q0,  -2*qx]], np.float32)

def dR_by_dqy(q):
    q0 = float(q[0])
    qx = float(q[1])
    qy = float(q[2])
    qz = float(q[3])

    return np.matrix([
        [-2*qy, 2*qx,  2*q0],
        [2*qx, 2*qy,  2*qz],
        [-2*q0, 2*qz, -2*qy]], np.float32)

def dR_by_dqz(q):
    q0 = float(q[0])
    qx = float(q[1])
    qy = float(q[2])
    qz = float(q[3])

    return np.matrix([
        [-2*qz, -2*q0, 2*qx],
        [2*q0, -2*qz, 2*qy],
        [2*qx,  2*qy, 2*qz]], np.float32)


def compute_F_cam(x, dt):
    omega = x[10:13]
    omega_tmp=x[10:13] * dt
    q_old=x[3:7]
    F = np.matrix(np.eye(13))
    qwt=quaternions.v2q((omega_tmp[0], omega_tmp[1], omega_tmp[2]))

    F[3:7, 3:7] = dq3_dq2(qwt)
    F[0:3, 7:10] = np.eye(3) * dt
    F[3:7, 10:13] = dq3_dq1(q_old) * domega_dt(omega, dt)
    return F

def domega_dt(o, dt):
    a = float(o[0])
    b = float(o[1])
    c = float(o[2])
    o_mod = np.linalg.norm(o)
    return np.matrix([
        [q0_oA(a, o_mod, dt)      , q0_oA(b, o_mod, dt)      , q0_oA(c, o_mod, dt)      ],
        [qA_oA(a, o_mod, dt)      , qA_oB(a, b, o_mod, dt), qA_oB(a, c, o_mod, dt)],
        [qA_oB(b, a, o_mod, dt), qA_oA(b, o_mod, dt)      , qA_oB(b, c, o_mod, dt)],
        [qA_oB(c, a, o_mod, dt), qA_oB(c, b, o_mod, dt), qA_oA(c, o_mod, dt)      ],
    ])

def q0_oA(o, m, dt):
    return (-dt / 2.0) * (o / m) * sin(m * dt / 2.0)

def qA_oB(o1, o2, m, dt):
    return (o1 * o2 / (m * m)) * ((dt / 2.0) * cos(m * dt / 2.0) - (1.0 / m) * sin(m * dt / 2.0))

def qA_oA(o, m, dt):
    return (dt / 2.0) * o * o / (m * m) * cos(m * dt / 2.0) + (1.0 / m) * (1.0 - o * o / (m * m)) * sin(m * dt / 2.0)

def dq3_dq1(q):
    R = float(q[0])
    X = float(q[1])
    Y = float(q[2])
    Z = float(q[3])

    return np.matrix([
        [R, -X, -Y, -Z],
        [X,  R, -Z,  Y],
        [Y,  Z,  R, -X],
        [Z, -Y,  X,  R]
    ], np.float32)

def dq3_dq2(q):
    R = float(q[0])
    X = float(q[1])
    Y = float(q[2])
    Z = float(q[3])

    return np.matrix([
        [R, -X, -Y, -Z],
        [X,  R,  Z, -Y],
        [Y, -Z,  R,  X],
        [Z,  Y, -X,  R]
    ], np.float32)


def calculate_H_inverse_depth(x_camera, y, K, h, features_size, feature_cur_position):
    H = np.zeros((2, 13 + 6 * features_size), np.float32)
    H[:, 0:13] = dh_dxv( K, x_camera, y, h)
    H[:,13 + 6 * (feature_cur_position): 13 + 6 * (feature_cur_position + 1)] = dh_dy( K, x_camera, y, h)

    return H


def dh_dy( camera, Xv_km1_k, yi, zi ):
    return dh_dhrl( camera, Xv_km1_k, yi, zi ) * dhrl_dy( Xv_km1_k, yi )




def dhrl_dy( x_camera, y ):
    rw = x_camera[0:3]
    Rrw = quaternions.q2r(x_camera[3:7]).I
    l = float(y[5])
    phi = float(y[4])
    theta = float(y[3])

    dmi_dthetai = Rrw * np.matrix([[cos(phi)*cos(theta), 0, -cos(phi)*sin(theta)]], np.float32).T
    dmi_dphii = Rrw * np.matrix([[-sin(phi)*sin(theta), -cos(phi), -sin(phi)*cos(theta)]], np.float32).T

    return np.concatenate([l*Rrw, dmi_dthetai, dmi_dphii, Rrw*(y[0:3]-rw)], axis=1)


def dh_dxv( camera, Xv_km1_k, y, zi ):
    return np.concatenate([ dh_drw( camera, Xv_km1_k, y, zi ), dh_dqwr( camera, Xv_km1_k, y, zi ), np.zeros((2, 6), np.float32)], axis=1)




def dh_dqwr( camera, Xv_km1_k, yi, zi ):
    return dh_dhrl( camera, Xv_km1_k, yi, zi ) * dhrl_dqwr( Xv_km1_k, yi )




def dhrl_dqwr( Xv_km1_k, yi ):

    rw = Xv_km1_k[0:3]
    qwr = Xv_km1_k[3:7]
    l = float(yi[5])
    phi = float(yi[4])
    theta = float(yi[3])
    mi = np.matrix([[cos(phi)*sin(theta), -sin(phi), cos(phi)*cos(theta)]], np.float32).T

    return dRq_times_a_by_dq(quaternions.qconj(qwr), (yi[0:3] - rw) * l + mi) * dqbar_by_dq()

def dh_drw( camera, Xv_km1_k, yi, zi ):
    return dh_dhrl( camera, Xv_km1_k, yi, zi ) * dhrl_drw( Xv_km1_k, yi )


def dhrl_drw( Xv_km1_k, yi ):
    return -(quaternions.q2r( Xv_km1_k[3:7] ).I ) * float(yi[5])


def dh_dhrl( camera, Xv_km1_k, yi, zi ):
    return dhd_dhu( camera, zi )*dhu_dhrl( camera, Xv_km1_k, yi )


def dhd_dhu( camera, zi_d ):
    #inv_a = jacob_undistor_fm( camera, zi_d );
    return np.eye(2,2, dtype=np.float32)


def dhu_dhrl( camera, Xv_km1_k, yi ):
    f = camera[0, 0]
    ku = 1
    kv = 1
    rw = Xv_km1_k[0:3]
    Rrw = quaternions.q2r(Xv_km1_k[3:7]).I

    theta = float(yi[3])
    phi = float(yi[4])
    rho = float(yi[5])
    mi = np.matrix([[cos(phi)*sin(theta), -sin(phi), cos(phi)*cos(theta)]], np.float32).T

    hc = Rrw*((yi[0:3] - rw)*rho + mi)
    hcx = float(hc[0])
    hcy = float(hc[1])
    hcz = float(hc[2])
    return np.concatenate([np.matrix([[f*ku/(hcz), 0, -hcx*f*ku/(hcz ** 2)]], np.float32),
        np.matrix([[0, f*kv/(hcz), -hcy*f*kv/(hcz ** 2)]], np.float32)], axis=0)

def dqbar_by_dq():
    return np.matrix(np.diag([1, -1, -1, -1]), np.float32);
