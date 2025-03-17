from pylie import SO3
import numpy as np
from scipy.linalg import expm

gravity_vector = 9.81 * np.reshape([0, 0, 1], (3, 1))
G_matrix = np.block([
    [np.zeros((3, 3)), gravity_vector, np.zeros((3, 1))],
    [np.zeros((2, 5))]
])
N_matrix = np.block([
    [np.zeros((3, 3)), np.zeros((3, 2))],
    [np.zeros((1, 3)), 0, -1],
    [np.zeros((1, 3)), 0, 0]
])

def calcposNED(lat, lng, alt, lat_ref, lng_ref, alt_ref):
    # This is a reimplementation of the ardupilot c++ function "get_distance_NED_double".
    # Importantly, we use the same constants.
    def longitude_scale(lat):
        DEG_TO_RAD = np.pi/180.
        scale = np.cos(lat * DEG_TO_RAD);
        return max(scale, 0.01);
    LOCATION_SCALING_FACTOR = 0.011131884502145034 * 1.e7
    diff = np.reshape(((lat - lat_ref) * LOCATION_SCALING_FACTOR,
                    (lng-lng_ref) * LOCATION_SCALING_FACTOR * longitude_scale((lat_ref+lat)/2.),
                    (alt_ref - alt)), (3,1))
    return diff


def SIM23_inv(Z):
    ZI = np.zeros((5,5))
    ZI[0:3,0:3] = Z[0:3,0:3].T
    AZ_inv = np.linalg.inv(Z[3:5,3:5])
    ZI[3:5,3:5] = AZ_inv
    ZI[0:3,3:5] = -Z[0:3,0:3].T @ Z[0:3,3:5] @ AZ_inv
    return ZI


class ComplementaryINS:
    Cp = np.array([[0],[1]])
    Cv = np.array([[1],[0]])
    def __init__(self, gain_kp=5.0, gain_kc=0.1, gain_Kq = np.eye(2), gain_kv=5.0, gain_kd=0.1, gain_km=0.1, gain_A0=np.eye(2)):
        self.XHat = np.identity(5)
        self.ZHat = np.identity(5)

        self.ZHat[3:5,3:5] = gain_A0

        self.gain_kp = gain_kp
        self.gain_kc = gain_kc
        self.gain_Kq = gain_Kq

        self.gain_kv = gain_kv
        self.gain_kd = gain_kd
        
        self.gain_km = gain_km

        self.saved_gps_pos = np.zeros((3,1))
        self.saved_gps_vel = np.zeros((3,1))
        self.saved_mag = np.zeros((3,1))
        self.saved_mag_ref = np.zeros((3,1))

    def set_IC(self, XHat_IC):
        self.XHat = XHat_IC
        self.ZHat[0:3,3:5] = XHat_IC[0:3,3:5] @ self.ZHat[3:5,3:5]

    def compass_update(self, m, m0):
        self.saved_mag_ref = m0
        self.saved_mag = m

    def compute_corrections(self, pos_true, vel_true, m0, m):
        yHat = self.XHat[0:3,4:5]
        yvHat = self.XHat[0:3,3:4]
        V_Z = self.ZHat[0:3, 3:5]
        A_Z = self.ZHat[3:5, 3:5]
        R_Z = self.ZHat[0:3, 0:3]

        Cp = ComplementaryINS.Cp
        Cv = ComplementaryINS.Cv

        A_Z_inv = np.linalg.inv(A_Z)
        y_Z = V_Z @ A_Z_inv @ Cp
        yv_Z = V_Z @ A_Z_inv @ Cv

        Omega_Delta = 4*self.gain_kc * R_Z.T @ SO3.skew(yHat - y_Z) @ (pos_true - y_Z) \
            + 4*self.gain_kd * R_Z.T @ SO3.skew(yvHat - yv_Z) @ (vel_true - yv_Z) \
            -self.gain_km * R_Z.T @ SO3.skew(m0) @ self.XHat[0:3,0:3] @ m

        W_Delta = (self.gain_kp+self.gain_kc) * R_Z.T @ (pos_true - yHat) @ Cp.T  @ A_Z_inv.T \
            + (self.gain_kv+self.gain_kd) * R_Z.T @ (vel_true - yvHat) @ Cv.T  @ A_Z_inv.T

        Omega_Gamma = np.zeros((3,1))
        W_Gamma = -(self.gain_kp+self.gain_kc) * R_Z.T @ (pos_true - y_Z) @ Cp.T @ A_Z_inv.T \
            -(self.gain_kv+self.gain_kd) * R_Z.T @ (vel_true - yv_Z) @ Cv.T @ A_Z_inv.T

        S_Gamma = -0.5*self.gain_kp * A_Z_inv @ Cp @ Cp.T @ A_Z_inv.T \
            -0.5*self.gain_kv * A_Z_inv @ Cv @ Cv.T @ A_Z_inv.T \
            + 0.5 * A_Z.T @ self.gain_Kq @ A_Z
        
        # Triangular S_Gamma
        # S_Gamma = np.triu(S_Gamma) + np.diag(np.diag(S_Gamma))

        Delta = np.zeros((5,5))
        Delta[0:3,0:3] = SO3.skew(Omega_Delta)
        Delta[0:3,3:5] = W_Delta
        Gamma = np.zeros((5,5))
        Gamma[0:3,0:3] = SO3.skew(Omega_Gamma)
        Gamma[0:3,3:5] = W_Gamma
        Gamma[3:5,3:5] = S_Gamma

        return Delta, Gamma

    def integrate_dynamics(self, gyr, acc , dt):
        U = np.zeros((5,5))
        U[0:3,0:3] = SO3.skew(gyr)
        U[0:3,3:4] = acc

        Delta, Gamma = self.compute_corrections(self.saved_gps_pos, self.saved_gps_vel, self.saved_mag_ref, self.saved_mag)
        Z_inv = SIM23_inv(self.ZHat)
        DeltaBar = self.ZHat @ Delta @ Z_inv

        self.XHat = expm(dt * (G_matrix + N_matrix + DeltaBar)) @ self.XHat @ expm(dt * (U - N_matrix))
        self.ZHat = expm(dt * (G_matrix + N_matrix)) @ self.ZHat @ expm(-dt * Gamma)

        return Delta, Gamma
    
    def GPS_update(self, gps_pos, gps_vel):
        self.saved_gps_pos = gps_pos
        self.saved_gps_vel = gps_vel

    def state_estimates(self):
        # Return Attitude, Velocity, Position estimates.
        euler_angles = SO3(self.XHat[0:3,0:3]).as_euler()
        if euler_angles[2] < 0.0:
            euler_angles[2] += 360.0
        return euler_angles, self.XHat[0:3,3], self.XHat[0:3,4]

class SimpleComplementaryINS:
    def __init__(self):
        self.RHat = SO3.identity()
        self.pHat = np.zeros((3,1))
        self.vHat = np.zeros((3,1))

        self.pAux = np.zeros((3,1))
        self.vAux = np.zeros((3,1))

        self.gain_r = 2.0
        self.gain_c = 0.0005
        self.gain_l = 1.0
        self.GPS_error_threshold = 1e2

        self.gain_m = 0*1e-5

        self.saved_gps_pos = np.zeros((3,1))
        self.saved_mag = np.zeros((3,1))
        self.saved_mag_ref = np.zeros((3,1))

    def compass_update(self, m, m0):
        self.saved_mag_ref = m0
        self.saved_mag = m

    def compute_corrections(self, pos_true, m0, m):

        if np.linalg.norm(pos_true - self.pHat) > self.GPS_error_threshold:
            pos_true = self.pHat.copy()
        
        Omega_Delta = 2*self.gain_c * SO3.skew(self.pHat - self.pAux) @ (pos_true - self.pAux) \
                        - self.gain_m * SO3.skew(m0) @ (self.RHat @ m)
        w_p_Delta = (self.gain_c+self.gain_r) * (pos_true - self.pHat)
        w_v_Delta = self.gain_l * w_p_Delta

        w_p_Gamma = (self.gain_c+self.gain_r) * (pos_true - self.pAux)
        w_v_Gamma = self.gain_l * w_p_Gamma

        return Omega_Delta, w_p_Delta, w_v_Delta, w_p_Gamma, w_v_Gamma

    def integrate_dynamics(self, gyr, acc , dt):
        Omega_Delta, w_p_Delta, w_v_Delta, w_p_Gamma, w_v_Gamma = self.compute_corrections(self.saved_gps_pos, self.saved_mag_ref, self.saved_mag)

        # Integrate
        self.pHat += dt*(self.vHat + w_p_Delta + SO3.skew(Omega_Delta) @ (self.pHat -self.pAux))
        self.vHat += dt*(self.RHat @ acc_est + gravity_vector + w_v_Delta  + SO3.skew(Omega_Delta) @ (self.vHat -self.vAux))
        self.RHat = SO3.exp(dt * Omega_Delta) * self.RHat * SO3.exp(dt * gyr_est)

        self.pAux += dt*(self.vAux + w_p_Gamma)
        self.vAux += dt*(gravity_vector + w_v_Gamma)
    
    def GPS_update(self, gps_pos):
        self.saved_gps_pos = gps_pos

    def state_estimates(self):
        # Return Attitude, Velocity, Position estimates.
        euler_angles = self.RHat.as_euler() * 180.0 / np.pi
        if euler_angles[2] < 0.0:
            euler_angles[2] += 360.0
        return euler_angles, self.vHat.copy().ravel(), self.pHat.copy().ravel()