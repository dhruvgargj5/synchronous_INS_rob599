import numpy as np
from pylie import SO3


class Dimensions:
    @staticmethod
    def vec3():
        return ['x', 'y', 'z']
    @staticmethod
    def quat():
        return ['x', 'y', 'z', 'w']

def get_from_df(df, column_prefix, dims, step):
    # odom
    #   pose (x,y,z)
    #   orientation (x,y,z,w)
    #   twist
    #       linear velocity (x,y,z)
    #       angular velocity (x,y,z)
    # imu
    #   linear (x,y,z)
    #   angular (x,y,z)
    #   orientation (x,y,z,w)
    return df.iloc[step][[f"{column_prefix}_{d}" for d in dims]]
def get_true_state_from_df(df, step):
    X = np.eye(5)
    orientation_quat = get_from_df(df, "odom_orientation", Dimensions.quat(), step)
    X[:3,:3] = SO3.from_list(orientation_quat.tolist(), format_spec='q').as_matrix()
    X[0:3,3] = get_from_df(df, "odom_vel", Dimensions.vec3(), step).to_numpy().astype(float).reshape(3)
    X[0:3,4] = get_from_df(df, "odom_pose", Dimensions.vec3(), step).to_numpy().astype(float).reshape(3)
    return X
def get_noisy_state_from_df(df, step):
    X = np.eye(5)
    orientation_quat = get_from_df(df, "odom_orientation_noisy", Dimensions.quat(), step)
    X[:3,:3] = SO3.from_list(orientation_quat.tolist(), format_spec='q').as_matrix()
    X[0:3,3] = get_from_df(df, "odom_vel_noisy", Dimensions.vec3(), step).to_numpy().astype(float).reshape(3)
    X[0:3,4] = get_from_df(df, "odom_pose_noisy", Dimensions.vec3(), step).to_numpy().astype(float).reshape(3)
    return X

def rotations_from_quat(quat):
    rpys = []
    for q in quat:
        rpy = SO3.from_list(q, format_spec="q").as_matrix()
        rpys.append(rpy)
    return np.asarray(rpys)

def rpy_from_quat(quat):
    rpys = []
    for q in quat:
        rpy = SO3.from_list(q, format_spec="q").as_euler(
            seq="xyz", degrees=False
        )
        rpys.append(rpy)
    return np.asarray(rpys)
