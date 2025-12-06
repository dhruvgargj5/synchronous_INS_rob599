#!/usr/bin/env python3
"""
===================================================================================
EXTENDED KALMAN FILTER (EKF) - FIXED VERSION FOR YOUR CSV
===================================================================================

This version correctly matches your CSV columns from bag_to_csv.py:

IMU Data:
    - imu_angular_vel_x, imu_angular_vel_y, imu_angular_vel_z
    - imu_linear_acc_x, imu_linear_acc_y, imu_linear_acc_z
    - imu_orientation_x, imu_orientation_y, imu_orientation_z, imu_orientation_w

Ground Truth (Map Frame):
    - map_position_x, map_position_y, map_position_z
    - map_orientation_x, map_orientation_y, map_orientation_z, map_orientation_w

Odometry:
    - odom_pose_x, odom_pose_y, odom_pose_z
    - odom_orientation_x, odom_orientation_y, odom_orientation_z, odom_orientation_w
    - odom_twist_linear_x, odom_twist_linear_y, odom_twist_linear_z
    - odom_twist_angular_x, odom_twist_angular_y, odom_twist_angular_z

Command Velocity:
    - cmd_vel_linear_x, cmd_vel_linear_y, cmd_vel_linear_z
    - cmd_vel_angular_x, cmd_vel_angular_y, cmd_vel_angular_z

Author: For Rajat's 599 Project
"""

import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import os
import pandas as pd


# ===================================================================================
# HELPER FUNCTIONS
# ===================================================================================

def quaternion_to_euler(qx, qy, qz, qw):
    """
    Convert quaternion to euler angles (roll, pitch, yaw).
    
    PURE PYTHON IMPLEMENTATION - No scipy needed!
    """
    # Handle invalid quaternion
    if qx == 0 and qy == 0 and qz == 0 and qw == 0:
        return 0.0, 0.0, 0.0
    
    # Normalize quaternion
    norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm < 1e-10:
        return 0.0, 0.0, 0.0
    
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def normalize_angle(angle):
    """Keep angle between -π and +π."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        if value is None or value == '':
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


# ===================================================================================
# CSV READER
# ===================================================================================

def load_csv_data(csv_path):
    """Load CSV data using Python's built-in csv module."""
    print(f"Loading CSV: {csv_path}")
    
    rows = []
    columns = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    print(f"✓ Loaded {len(rows)} rows")
    print(f"✓ Found {len(columns)} columns")
    
    return columns, rows


# ===================================================================================
# EXTENDED KALMAN FILTER
# ===================================================================================

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for 3D Robot Pose Estimation.
    
    STATE VECTOR (15 states):
        x[0]  = x position
        x[1]  = y position
        x[2]  = z position
        x[3]  = roll
        x[4]  = pitch
        x[5]  = yaw
        x[6]  = velocity x
        x[7]  = velocity y
        x[8]  = velocity z
        x[9]  = angular velocity x
        x[10] = angular velocity y
        x[11] = angular velocity z
        x[12] = accel bias x
        x[13] = accel bias y
        x[14] = accel bias z
    """
    
    def __init__(self):
        self.n_states = 15
        self.x = np.zeros(self.n_states)
        
        # Covariance matrix
        self.P = np.eye(self.n_states)
        self.P[0, 0] = 1.0
        self.P[1, 1] = 1.0
        self.P[2, 2] = 0.1
        self.P[3, 3] = 0.1
        self.P[4, 4] = 0.1
        self.P[5, 5] = 0.5
        self.P[6, 6] = 0.5
        self.P[7, 7] = 0.5
        self.P[8, 8] = 0.1
        self.P[9, 9] = 0.1
        self.P[10, 10] = 0.1
        self.P[11, 11] = 0.1
        self.P[12, 12] = 0.1
        self.P[13, 13] = 0.1
        self.P[14, 14] = 0.1
        
        # Process noise
        self.Q = np.diag([
            0.001, 0.001, 0.0001,  # position
            0.001, 0.001, 0.001,   # orientation
            0.01, 0.01, 0.001,     # velocity
            0.01, 0.01, 0.01,      # angular velocity
            0.0001, 0.0001, 0.0001 # biases
        ])
        
        # IMU measurement noise
        self.R_imu = np.diag([0.001, 0.001, 0.001, 0.05, 0.05, 0.05])
        
        # Position measurement noise
        self.R_pos = np.diag([0.01, 0.01, 0.001, 0.01, 0.01, 0.02])
        
        self.initialized = False
        self.gravity = 9.81
    
    def initialize_state(self, pos_x, pos_y, pos_z, roll, pitch, yaw, vx=0, vy=0, vz=0):
        """Set initial state."""
        self.x[0] = pos_x
        self.x[1] = pos_y
        self.x[2] = pos_z
        self.x[3] = roll
        self.x[4] = pitch
        self.x[5] = yaw
        self.x[6] = vx
        self.x[7] = vy
        self.x[8] = vz
        self.x[9:15] = 0.0
        self.initialized = True
        print(f"✓ EKF Initialized at ({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f})")
        print(f"  Orientation: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")
    
    def predict(self, dt):
        """Prediction step."""
        if not self.initialized or dt <= 0:
            return
        
        px, py, pz = self.x[0], self.x[1], self.x[2]
        roll, pitch, yaw = self.x[3], self.x[4], self.x[5]
        vx, vy, vz = self.x[6], self.x[7], self.x[8]
        wx, wy, wz = self.x[9], self.x[10], self.x[11]
        
        # Rotation matrix
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        
        # Predict position
        vel_world = R @ np.array([vx, vy, vz])
        self.x[0] = px + vel_world[0] * dt
        self.x[1] = py + vel_world[1] * dt
        self.x[2] = pz + vel_world[2] * dt
        
        # Predict orientation
        self.x[3] = normalize_angle(roll + wx * dt)
        self.x[4] = normalize_angle(pitch + wy * dt)
        self.x[5] = normalize_angle(yaw + wz * dt)
        
        # Jacobian
        F = np.eye(self.n_states)
        F[0, 6] = R[0, 0] * dt
        F[0, 7] = R[0, 1] * dt
        F[0, 8] = R[0, 2] * dt
        F[1, 6] = R[1, 0] * dt
        F[1, 7] = R[1, 1] * dt
        F[1, 8] = R[1, 2] * dt
        F[2, 6] = R[2, 0] * dt
        F[2, 7] = R[2, 1] * dt
        F[2, 8] = R[2, 2] * dt
        F[3, 9] = dt
        F[4, 10] = dt
        F[5, 11] = dt
        
        self.P = F @ self.P @ F.T + self.Q * dt
    
    def update_imu(self, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z):
        """Update with IMU measurements."""
        if not self.initialized:
            return
        
        z = np.array([gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z])
        
        roll, pitch = self.x[3], self.x[4]
        expected_gyro = np.array([self.x[9], self.x[10], self.x[11]])
        expected_accel = np.array([
            -self.gravity * np.sin(pitch) + self.x[12],
            self.gravity * np.cos(pitch) * np.sin(roll) + self.x[13],
            self.gravity * np.cos(pitch) * np.cos(roll) + self.x[14]
        ])
        z_expected = np.concatenate([expected_gyro, expected_accel])
        
        H = np.zeros((6, self.n_states))
        H[0, 9] = 1.0
        H[1, 10] = 1.0
        H[2, 11] = 1.0
        H[3, 4] = -self.gravity * np.cos(pitch)
        H[3, 12] = 1.0
        H[4, 3] = self.gravity * np.cos(pitch) * np.cos(roll)
        H[4, 4] = -self.gravity * np.sin(pitch) * np.sin(roll)
        H[4, 13] = 1.0
        H[5, 3] = -self.gravity * np.cos(pitch) * np.sin(roll)
        H[5, 4] = -self.gravity * np.sin(pitch) * np.cos(roll)
        H[5, 14] = 1.0
        
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)
        
        y = z - z_expected
        self.x = self.x + K @ y
        self.x[3] = normalize_angle(self.x[3])
        self.x[4] = normalize_angle(self.x[4])
        self.x[5] = normalize_angle(self.x[5])
        
        self.P = (np.eye(self.n_states) - K @ H) @ self.P
    
    def update_position(self, pos_x, pos_y, pos_z, roll, pitch, yaw):
        """Update with position measurements."""
        if not self.initialized:
            return
        
        z = np.array([pos_x, pos_y, pos_z, roll, pitch, yaw])
        z_expected = np.array([
            self.x[0], self.x[1], self.x[2],
            self.x[3], self.x[4], self.x[5]
        ])
        
        H = np.zeros((6, self.n_states))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0
        H[4, 4] = 1.0
        H[5, 5] = 1.0
        
        S = H @ self.P @ H.T + self.R_pos
        K = self.P @ H.T @ np.linalg.inv(S)
        
        y = z - z_expected
        y[3] = normalize_angle(y[3])
        y[4] = normalize_angle(y[4])
        y[5] = normalize_angle(y[5])
        
        self.x = self.x + K @ y
        self.x[3] = normalize_angle(self.x[3])
        self.x[4] = normalize_angle(self.x[4])
        self.x[5] = normalize_angle(self.x[5])
        
        self.P = (np.eye(self.n_states) - K @ H) @ self.P
    
    def get_state(self):
        return self.x.copy()
    
    def get_covariance(self):
        return self.P.copy()


# ===================================================================================
# MAIN PROCESSING
# ===================================================================================

def process_csv_with_ekf(csv_path):
    """Process CSV and run EKF."""
    
    columns, rows = load_csv_data(csv_path)
    
    # Print some column names to verify
    print(f"\nColumn names (first 20):")
    for i, col in enumerate(columns[:20]):
        print(f"  {i}: '{col}'")
    
    # Check for key columns
    print(f"\nChecking for required columns...")
    required = ['timestamp', 'imu_angular_vel_x', 'map_position_x', 'odom_pose_x']
    for col in required:
        if col in columns:
            print(f"  ✓ Found '{col}'")
        else:
            print(f"  ✗ Missing '{col}'")
    
    # Create EKF
    ekf = ExtendedKalmanFilter()
    
    # Results storage
    results = {
        'time': [],
        'true_x': [], 'true_y': [], 'true_z': [],
        'true_roll': [], 'true_pitch': [], 'true_yaw': [],
        'true_vx': [], 'true_vy': [], 'true_vz': [],
        'est_x': [], 'est_y': [], 'est_z': [],
        'est_roll': [], 'est_pitch': [], 'est_yaw': [],
        'est_vx': [], 'est_vy': [], 'est_vz': [],
        'error_x': [], 'error_y': [], 'error_z': [],
        'error_roll': [], 'error_pitch': [], 'error_yaw': [],
        'error_position': [], 'error_attitude': [],
        'cov_x': [], 'cov_y': [], 'cov_yaw': []
    }
    
    prev_time = None
    init_attempts = 0
    
    print(f"\nProcessing {len(rows)} data points...")
    
    for idx, row in enumerate(rows):
        # === GET TIMESTAMP ===
        current_time = safe_float(row.get('timestamp', 0), idx * 0.01)
        
        # === GET IMU DATA ===
        gyro_x = safe_float(row.get('imu_angular_vel_x', 0))
        gyro_y = safe_float(row.get('imu_angular_vel_y', 0))
        gyro_z = safe_float(row.get('imu_angular_vel_z', 0))
        accel_x = safe_float(row.get('imu_linear_acc_x', 0))
        accel_y = safe_float(row.get('imu_linear_acc_y', 0))
        accel_z = safe_float(row.get('imu_linear_acc_z', 0))
        
        # === GET GROUND TRUTH FROM MAP FRAME ===
        map_x = safe_float(row.get('map_position_x', 0))
        map_y = safe_float(row.get('map_position_y', 0))
        map_z = safe_float(row.get('map_position_z', 0))
        map_qx = safe_float(row.get('map_orientation_x', 0))
        map_qy = safe_float(row.get('map_orientation_y', 0))
        map_qz = safe_float(row.get('map_orientation_z', 0))
        map_qw = safe_float(row.get('map_orientation_w', 1))
        
        # === GET ODOMETRY DATA (use as ground truth if map is zeros) ===
        odom_x = safe_float(row.get('odom_pose_x', 0))
        odom_y = safe_float(row.get('odom_pose_y', 0))
        odom_z = safe_float(row.get('odom_pose_z', 0))
        odom_qx = safe_float(row.get('odom_orientation_x', 0))
        odom_qy = safe_float(row.get('odom_orientation_y', 0))
        odom_qz = safe_float(row.get('odom_orientation_z', 0))
        odom_qw = safe_float(row.get('odom_orientation_w', 1))
        
        # Velocity from odometry
        odom_vx = safe_float(row.get('odom_twist_linear_x', 0))
        odom_vy = safe_float(row.get('odom_twist_linear_y', 0))
        odom_vz = safe_float(row.get('odom_twist_linear_z', 0))
        
        # === DECIDE WHICH POSITION TO USE AS GROUND TRUTH ===
        # If map position is all zeros, use odometry instead
        if map_x == 0 and map_y == 0 and map_z == 0:
            # Use odometry as ground truth
            true_x, true_y, true_z = odom_x, odom_y, odom_z
            true_qx, true_qy, true_qz, true_qw = odom_qx, odom_qy, odom_qz, odom_qw
        else:
            # Use map frame as ground truth
            true_x, true_y, true_z = map_x, map_y, map_z
            true_qx, true_qy, true_qz, true_qw = map_qx, map_qy, map_qz, map_qw
        
        # Convert quaternion to euler
        true_roll, true_pitch, true_yaw = quaternion_to_euler(true_qx, true_qy, true_qz, true_qw)
        
        # === INITIALIZE EKF ===
        if not ekf.initialized:
            init_attempts += 1
            
            # Check if we have valid position data
            # Accept if either x or y is non-zero
            if true_x != 0 or true_y != 0:
                ekf.initialize_state(
                    true_x+0*true_x, true_y+0*true_y, true_z+0*true_z,
                    true_roll, true_pitch, true_yaw+0*true_yaw,
                    odom_vx+0*odom_vx, odom_vy+0*odom_vy, odom_vz+0*odom_vz  
                )
                prev_time = current_time
                print(f"  (Initialized at row {idx})")
                continue
            else:
                # Print debug info every 1000 attempts
                if init_attempts % 1000 == 0:
                    print(f"  Waiting for valid data... (row {idx}, map=({map_x},{map_y}), odom=({odom_x},{odom_y}))")
                continue
        
        # === CALCULATE TIME STEP ===
        dt = current_time - prev_time
        prev_time = current_time
        
        # Skip invalid time steps
        if dt <= 0 or dt > 1.0:
            continue
        
        # === EKF PREDICT ===
        ekf.predict(dt)
        
        # === EKF UPDATE WITH IMU ===
        ekf.update_imu(gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z)
        
        # === EKF UPDATE WITH POSITION (every 10 steps) ===
        if idx % 10 == 0 and (true_x != 0 or true_y != 0):
            ekf.update_position(true_x, true_y, true_z, true_roll, true_pitch, true_yaw)
        
        # === GET ESTIMATES ===
        state = ekf.get_state()
        cov = ekf.get_covariance()
        
        est_x, est_y, est_z = state[0], state[1], state[2]
        est_roll, est_pitch, est_yaw = state[3], state[4], state[5]
        est_vx, est_vy, est_vz = state[6], state[7], state[8]
        
        # === CALCULATE ERRORS ===
        error_x = est_x - true_x
        error_y = est_y - true_y
        error_z = est_z - true_z
        error_roll = normalize_angle(est_roll - true_roll)
        error_pitch = normalize_angle(est_pitch - true_pitch)
        error_yaw = normalize_angle(est_yaw - true_yaw)
        error_position = math.sqrt(error_x**2 + error_y**2 + error_z**2)
        error_attitude = math.sqrt(error_roll**2 + error_pitch**2 + error_yaw**2)
        
        # === STORE RESULTS ===
        results['time'].append(current_time)
        results['true_x'].append(true_x)
        results['true_y'].append(true_y)
        results['true_z'].append(true_z)
        results['true_roll'].append(np.degrees(true_roll))
        results['true_pitch'].append(np.degrees(true_pitch))
        results['true_yaw'].append(np.degrees(true_yaw))
        results['true_vx'].append(odom_vx)
        results['true_vy'].append(odom_vy)
        results['true_vz'].append(odom_vz)
        results['est_x'].append(est_x)
        results['est_y'].append(est_y)
        results['est_z'].append(est_z)
        results['est_roll'].append(np.degrees(est_roll))
        results['est_pitch'].append(np.degrees(est_pitch))
        results['est_yaw'].append(np.degrees(est_yaw))
        results['est_vx'].append(est_vx)
        results['est_vy'].append(est_vy)
        results['est_vz'].append(est_vz)
        results['error_x'].append(error_x)
        results['error_y'].append(error_y)
        results['error_z'].append(error_z)
        results['error_roll'].append(np.degrees(error_roll))
        results['error_pitch'].append(np.degrees(error_pitch))
        results['error_yaw'].append(np.degrees(error_yaw))
        results['error_position'].append(error_position)
        results['error_attitude'].append(np.degrees(error_attitude))
        results['cov_x'].append(np.sqrt(cov[0, 0]))
        results['cov_y'].append(np.sqrt(cov[1, 1]))
        results['cov_yaw'].append(np.degrees(np.sqrt(cov[5, 5])))
        
        # Progress update
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(rows)} rows...")
    
    print(f"✓ Processed {len(results['time'])} valid samples")
    
    return results


# ===================================================================================
# PLOTTING
# ===================================================================================

import math
import numpy as np
import matplotlib.pyplot as plt

def quaternion_to_euler(qx, qy, qz, qw):
    """
    Convert quaternion to euler angles (roll, pitch, yaw) in *radians*.
    """
    # Handle invalid quaternion
    if qx == 0 and qy == 0 and qz == 0 and qw == 0:
        return 0.0, 0.0, 0.0

    # Normalize quaternion
    norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm < 1e-10:
        return 0.0, 0.0, 0.0

    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def create_plots(results, ground_truth, save_prefix='ekf'):
    """Create all 4 figures comparing ground truth vs EKF estimates."""

    if len(results['time']) == 0:
        print("No data to plot!")
        return

    # Helper to fetch a ground-truth column from ground_truth with fallbacks
    def get_truth(gt_source, *candidate_names, fallback=None):
        """
        Try each candidate name in order; if found in gt_source (dict or DataFrame),
        return it as a numpy array. If none found, fall back to results[fallback].
        """
        for name in candidate_names:
            if isinstance(gt_source, dict) and name in gt_source:
                return np.asarray(gt_source[name])
            if hasattr(gt_source, "__contains__") and name in gt_source:
                return np.asarray(gt_source[name])
        if fallback is not None:
            return np.asarray(results[fallback])
        raise KeyError(
            f"None of {candidate_names} found in ground_truth and no fallback provided."
        )

    # Time vector
    time = np.asarray(results['time'])
    time = time - time[0]  # Start from 0

    print("\nCreating plots...")

    # ---------------------------------------------------------------------
    # Ground-truth POSITION and VELOCITY from CSV
    # ---------------------------------------------------------------------
    true_x = get_truth(ground_truth, 'true_x', 'odom_pose_x', fallback='true_x')
    true_y = get_truth(ground_truth, 'true_y', 'odom_pose_y', fallback='true_y')
    true_z = get_truth(ground_truth, 'true_z', 'odom_pose_z', fallback='true_z')

    true_vx = get_truth(
        ground_truth, 'true_vx', 'odom_vel_x', 'odom_twist_x', fallback='true_vx'
    )
    true_vy = get_truth(
        ground_truth, 'true_vy', 'odom_vel_y', 'odom_twist_y', fallback='true_vy'
    )
    true_vz = get_truth(
        ground_truth, 'true_vz', 'odom_vel_z', 'odom_twist_z', fallback='true_vz'
    )

    # ---------------------------------------------------------------------
    # Ground-truth ATTITUDE from odom quaternions
    #   odom_orientation_x / y / z / w  -> roll, pitch, yaw (deg)
    # ---------------------------------------------------------------------
    if all(name in ground_truth for name in [
        'odom_orientation_x', 'odom_orientation_y',
        'odom_orientation_z', 'odom_orientation_w'
    ]):
        qx = np.asarray(ground_truth['odom_orientation_x'])
        qy = np.asarray(ground_truth['odom_orientation_y'])
        qz = np.asarray(ground_truth['odom_orientation_z'])
        qw = np.asarray(ground_truth['odom_orientation_w'])

        roll_list = []
        pitch_list = []
        yaw_list = []
        for i in range(len(qx)):
            r, p, y = quaternion_to_euler(qx[i], qy[i], qz[i], qw[i])
            roll_list.append(r)
            pitch_list.append(p)
            yaw_list.append(y)

        # Convert to degrees for plotting / comparison
        true_roll  = np.rad2deg(np.asarray(roll_list))
        true_pitch = np.rad2deg(np.asarray(pitch_list))
        true_yaw   = np.rad2deg(np.asarray(yaw_list))
    else:
        # Fallback if quaternions not available
        true_roll  = get_truth(ground_truth, 'true_roll', 'roll', fallback='true_roll')
        true_pitch = get_truth(ground_truth, 'true_pitch', 'pitch', fallback='true_pitch')
        true_yaw   = get_truth(ground_truth, 'true_yaw', 'yaw', fallback='true_yaw')

    # ---------------------------------------------------------------------
    # EKF estimates (assumed already in deg for attitude, m for pos, m/s for vel)
    # ---------------------------------------------------------------------
    est_x = np.asarray(results['est_x'])
    est_y = np.asarray(results['est_y'])
    est_z = np.asarray(results['est_z'])

    est_vx = np.asarray(results['est_vx'])
    est_vy = np.asarray(results['est_vy'])
    est_vz = np.asarray(results['est_vz'])

    est_roll  = np.asarray(results['est_roll'])
    est_pitch = np.asarray(results['est_pitch'])
    est_yaw   = np.asarray(results['est_yaw'])

    # ---------------------------------------------------------------------
    # ENFORCE CONSISTENT LENGTHS (important for scatter: x,y,c)
    # ---------------------------------------------------------------------
    arrays_to_sync = [
        time,
        true_x, true_y, true_z,
        true_vx, true_vy, true_vz,
        true_roll, true_pitch, true_yaw,
        est_x, est_y, est_z,
        est_vx, est_vy, est_vz,
        est_roll, est_pitch, est_yaw,
    ]
    min_len = min(len(a) for a in arrays_to_sync)

    time      = time[:min_len]
    true_x    = true_x[:min_len]
    true_y    = true_y[:min_len]
    true_z    = true_z[:min_len]
    true_vx   = true_vx[:min_len]
    true_vy   = true_vy[:min_len]
    true_vz   = true_vz[:min_len]
    true_roll = true_roll[:min_len]
    true_pitch= true_pitch[:min_len]
    true_yaw  = true_yaw[:min_len]

    est_x     = est_x[:min_len]
    est_y     = est_y[:min_len]
    est_z     = est_z[:min_len]
    est_vx    = est_vx[:min_len]
    est_vy    = est_vy[:min_len]
    est_vz    = est_vz[:min_len]
    est_roll  = est_roll[:min_len]
    est_pitch = est_pitch[:min_len]
    est_yaw   = est_yaw[:min_len]

    # Also trim error/cov arrays if needed
    results['error_attitude'] = np.asarray(results['error_attitude'])[:min_len]
    results['error_position'] = np.asarray(results['error_position'])[:min_len]
    results['error_x']        = np.asarray(results['error_x'])[:min_len]
    results['error_y']        = np.asarray(results['error_y'])[:min_len]
    results['error_z']        = np.asarray(results['error_z'])[:min_len]
    results['cov_x']          = np.asarray(results['cov_x'])[:min_len]
    results['cov_y']          = np.asarray(results['cov_y'])[:min_len]
    results['error_yaw']      = np.asarray(results['error_yaw'])[:min_len]

    # ========================================
    # FIGURE 1: XY TRAJECTORY
    # ========================================
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    scatter = ax1.scatter(
        true_x, true_y,
        c=time, cmap='viridis', s=3, label='Ground Truth'
    )
    ax1.plot(est_x, est_y, 'ro', linewidth=1, alpha=0.7, label='EKF Estimate')

    ax1.plot(true_x[0], true_y[0], 'go', ms=10, label='Start')
    ax1.plot(true_x[-1], true_y[-1], 'r*', ms=12, label='End')

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Time (s)')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Trajectory Colored by Time')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    plt.tight_layout()
    fig1.savefig(f'{save_prefix}_figure1_trajectory.png', dpi=150)
    print(f"  ✓ Saved {save_prefix}_figure1_trajectory.png")

    # ========================================
    # FIGURE 2: ESTIMATION (3x3)
    # ========================================
    fig2, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig2.suptitle('Attitude, Velocity, and Position Estimation', fontsize=14)

    # Attitude
    axes[0, 0].plot(time, true_roll, 'k-', lw=1.5, label='Ground Truth')
    axes[0, 0].plot(time, est_roll, 'r--', lw=1, label='EKF')
    axes[0, 0].set_ylabel('Roll (deg)')
    axes[0, 0].set_title('Attitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(time, true_pitch, 'k-', lw=1.5)
    axes[1, 0].plot(time, est_pitch, 'r--', lw=1)
    axes[1, 0].set_ylabel('Pitch (deg)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[2, 0].plot(time, true_yaw, 'k-', lw=1.5)
    axes[2, 0].plot(time, est_yaw, 'r--', lw=1)
    axes[2, 0].set_ylabel('Yaw (deg)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].grid(True, alpha=0.3)

    # Velocity
    axes[0, 1].plot(time, true_vx, 'k-', lw=1.5, label='Ground Truth')
    axes[0, 1].plot(time, est_vx, 'r--', lw=1, label='EKF')
    axes[0, 1].set_ylabel('Vx (m/s)')
    axes[0, 1].set_title('Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(time, true_vy, 'k-', lw=1.5)
    axes[1, 1].plot(time, est_vy, 'r--', lw=1)
    axes[1, 1].set_ylabel('Vy (m/s)')
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 1].plot(time, true_vz, 'k-', lw=1.5)
    axes[2, 1].plot(time, est_vz, 'r--', lw=1)
    axes[2, 1].set_ylabel('Vz (m/s)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True, alpha=0.3)

    # Position
    axes[0, 2].plot(time, true_x, 'k-', lw=1.5, label='Ground Truth')
    axes[0, 2].plot(time, est_x, 'r--', lw=1, label='EKF')
    axes[0, 2].set_ylabel('X (m)')
    axes[0, 2].set_title('Position')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 2].plot(time, true_y, 'k-', lw=1.5)
    axes[1, 2].plot(time, est_y, 'r--', lw=1)
    axes[1, 2].set_ylabel('Y (m)')
    axes[1, 2].grid(True, alpha=0.3)

    axes[2, 2].plot(time, true_z, 'k-', lw=1.5)
    axes[2, 2].plot(time, est_z, 'r--', lw=1)
    axes[2, 2].set_ylabel('Z (m)')
    axes[2, 2].set_xlabel('Time (s)')
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(f'{save_prefix}_figure2_estimation.png', dpi=150)
    print(f"  ✓ Saved {save_prefix}_figure2_estimation.png")

    # ========================================
    # FIGURE 3: ERROR METRICS
    # ========================================
    fig3, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig3.suptitle('Observer Error Metrics', fontsize=14)

    axes[0].semilogy(time, results['error_attitude'], 'b-', lw=1)
    axes[0].set_ylabel('Attitude\nError (deg)')
    axes[0].grid(True, alpha=0.3)

    vel_err = np.sqrt((est_vx - true_vx)**2 +
                      (est_vy - true_vy)**2)
    axes[1].semilogy(time, vel_err, 'g-', lw=1)
    axes[1].set_ylabel('Velocity\nError (m/s)')
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(time, results['error_position'], 'r-', lw=1)
    axes[2].set_ylabel('Position\nError (m)')
    axes[2].grid(True, alpha=0.3)

    total_cov = np.sqrt(np.array(results['cov_x'])**2 +
                        np.array(results['cov_y'])**2)
    total_cov = total_cov[:min_len]
    axes[3].semilogy(time, total_cov, 'm-', lw=1)
    axes[3].set_ylabel('Uncertainty (m)')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig(f'{save_prefix}_figure3_errors.png', dpi=150)
    print(f"  ✓ Saved {save_prefix}_figure3_errors.png")

    # ========================================
    # FIGURE 4: EXCITATION
    # ========================================
    fig4, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig4.suptitle('Velocity Excitation μᵥ and Position Excitation μₚ', fontsize=14)

    vel_err_x = est_vx - true_vx
    vel_err_y = est_vy - true_vy
    vel_err_z = est_vz - true_vz

    axes[0, 0].plot(time, vel_err_x, 'b-', lw=1)
    axes[0, 0].set_ylabel('X (m/s)')
    axes[0, 0].set_title('Velocity Excitation μᵥ')
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(time, vel_err_y, 'b-', lw=1)
    axes[1, 0].set_ylabel('Y (m/s)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[2, 0].plot(time, vel_err_z, 'b-', lw=1)
    axes[2, 0].set_ylabel('Z (m/s)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time, results['error_x'], 'r-', lw=1)
    axes[0, 1].set_ylabel('X (m)')
    axes[0, 1].set_title('Position Excitation μₚ')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(time, results['error_y'], 'r-', lw=1)
    axes[1, 1].set_ylabel('Y (m)')
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 1].plot(time, results['error_z'], 'r-', lw=1)
    axes[2, 1].set_ylabel('Z (m)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig4.savefig(f'{save_prefix}_figure4_excitation.png', dpi=150)
    print(f"  ✓ Saved {save_prefix}_figure4_excitation.png")

    plt.show()

    # Summary (unchanged)
    print("\n" + "="*50)
    print("EKF PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Position Error - Mean: {np.mean(results['error_position']):.4f} m")
    print(f"Position Error - Max:  {np.max(results['error_position']):.4f} m")
    print(f"Position Error - RMSE: {np.sqrt(np.mean(np.array(results['error_position'])**2)):.4f} m")
    print(f"Attitude Error - Mean: {np.mean(results['error_attitude']):.4f} deg")
    print(f"Yaw Error - Mean:      {np.mean(np.abs(results['error_yaw'])):.4f} deg")
    print("="*50)



# ===================================================================================
# MAIN
# ===================================================================================

def main():
    print("\n" + "="*60)
    print("  EXTENDED KALMAN FILTER - FIXED VERSION")
    print("="*60)
    
    # Find CSV file - EKF input
    csv_path = "D:\OneDrive - Umich\Courses\ROB 599 013 - Computational Symmetry in AI & Robotics\Final Project\Extended Kalman Filter\combined_data_30HZ_noisy.csv"
    
    if not os.path.exists(csv_path):
        print(f"\n⚠ '{csv_path}' not found.")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            csv_path = csv_files[0]
            print(f"Using: {csv_path}")
        else:
            print("No CSV files found!")
            return
    
    # Process
    results = process_csv_with_ekf(csv_path)
    
    if len(results['time']) == 0:
        print("\n⚠ No valid data processed!")
        print("Possible issues:")
        print("  1. map_position_x/y are all zeros (TF lookup failed)")
        print("  2. odom_pose_x/y are all zeros")
        print("  3. Column names don't match")
        return
    
    # Plot
    ground_truth_df = pd.read_csv("D:\OneDrive - Umich\Courses\ROB 599 013 - Computational Symmetry in AI & Robotics\Final Project\Extended Kalman Filter\combined_data_30Hz.csv")
    create_plots(results, ground_truth_df, save_prefix='ekf_run1')
    
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()