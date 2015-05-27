__author__ = 'Kirill'
import numpy as np


class Feature:
    def __init__(self):
        pass

    y = 0
    init_measurement = 0
    camera_pos_init = 0
    camera_rot_init = 0
    uv_init = 0
    init_frame = -1
    visible=True
    number=0
    index=0
    retrack=False
    z=0
    h=0
    H=0
    S=0
    R=np.eye(2, 2, dtype=np.float32)