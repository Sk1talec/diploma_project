import cv2
import numpy as np
import random

f = 2
c_x = 200
c_y = 200
K = np.array([
    [f, 0, c_x],
    [0, f, c_y],
    [0, 0,  1]
])

sigma_a = 0.007
sigma_alpha = 0.007
sigma_image_noise = 1.0

r_k_k = (0,0,0)

N_features = 20
x_camera = 12 # 3(3d pose) + 4( rotation quaternion ) + 3 ( linear velocity) + 2 ( angular velocity)
N = x_camera +  N_features * 6

x = np.matrix((1, N), np.float32) # State vector
u = np.array((1, 5), np.float32) # Control vector

P = np.identity(N, np.float32) # State covariance
z = np.matrix((10, 10)) # Measurement

F = np.matrix((10, 10)) # State transition Jacobian
H = np.matrix((10, 10)) # Measurement Jacobian


# These are given
Q = np.identity(N, np.float32) * sigma_alpha # Prediction error
R = np.identity(N, np.float32) * sigma_alpha # Measurement error
I = np.identity(10)

#state transition
def f(x, u):
    pass

#Observation
def h(x):
    pass

def readFrame(step):
    pass

# P_tmp is P[k, k - 1]
# P is either P[k - 1, k - 1] or P[k, k]

# x_tmp is x[k, k - 1]
# x is either x[k - 1, k - 1] or x[k, k]
def main():
    global P, x
    for k in xrange(1000):
        features = readFrame(k)

        # Predict
        x_tmp = f(x, u) #Predicted state estimate
        P_tmp = F[k - 1] * P * F[k - 1].t() + Q #Predicted covariance estimate

        #Update
        y = z - h(x_tmp)        # Innovation or measurement residual
        S = H[k] * P_tmp * H[k].t() + R  # Innovation (or residual) covariance
        K = P_tmp * H[k].t() * S.invert()     # Near-optimal Kalman gain
        x = x_tmp + K * y                # Updated state estimate
        P = (I - K * H[k]) * P_tmp      # Updated covariance estimate


if __name__ == '__main__':
    main()