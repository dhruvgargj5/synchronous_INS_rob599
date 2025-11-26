import numpy as np
from pylie import SE3, SO3, R3, SE23
import matplotlib.pyplot as plt
from scipy.linalg import expm
from dataclasses import dataclass, field
np.seterr('raise')

try:
    from progressbar import progressbar
except ImportError:
    progressbar = lambda x : x

from ins_observer import ComplementaryINS

plt.rc('lines', linewidth=1.0)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

np.random.seed(5992)

import pandas as pd
import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--animate', action='store_true')
parser.add_argument('--extreme-initial',action='store_true')

args = parser.parse_args()

class Dimensions:
    @staticmethod
    def vec3():
        return ['x', 'y', 'z']
    @staticmethod
    def quat():
        return ['x', 'y', 'z', 'w']

df = pd.read_csv('combined_data.csv')

sim_speed_multiplier : int = 10
time_lim = len(df)

if args.extreme_initial:
    print("The initial condition is set to extreme.")
else:
    print("The initial condition is set to standard.")

dt = 0.001 * sim_speed_multiplier
max_steps = int(time_lim)

figheight = 2.5
figsize_factor = 1.5

# Reference magnetic field
m0 = np.reshape((1.,0.,0.), (3,1))

# Set up observers
@dataclass
class ObserverInfo:
    name : str
    lc : str
    ls : str
    obs : ComplementaryINS
    states_est : list = field(default_factory=list)
    states_aux : list = field(default_factory=list)

# Standard gains
kp = 2.0
kv = 0.5
kc = 0.1
kd = 0.1
km = 2.0
Kq = np.diag((10.0,2.0))
A0 = np.diag((2.,10.))

print("dt: {}, t_max: {}".format(dt, time_lim))
print(("Kq: diag({},{}),"+"kp: {},"+"kc: {},"+"kv: {},"+"kd: {},"+"km: {}").format(Kq[0,0],Kq[1,1],kp,kc,kv,kd,km))
print(("\\diag({},{}) & {} & {} & {} & {} & {}").format(Kq[0,0],Kq[1,1],kp,kc,kv,kd,km))
print("A_Z(0): diag({},{})".format(A0[0,0], A0[1,1]))


observer_list = [
    ObserverInfo("Est. p", 'r', 'dotted', ComplementaryINS(gain_kp=kp, gain_kc=kc, gain_Kq = Kq, gain_kv=0.0, gain_kd=0.0, gain_km=0.0, gain_A0=A0)),
    # ObserverInfo("MEKF", 'b', 'dashed', MEKF()),
    ObserverInfo("Est. pv", 'g', 'dashed', ComplementaryINS(gain_kp=kp, gain_kc=kc, gain_Kq = Kq, gain_kv=kv, gain_kd=kd, gain_km=0.0, gain_A0=A0)),
    ObserverInfo("Est. pm", 'm', 'dashdot', ComplementaryINS(gain_kp=kp, gain_kc=kc, gain_Kq = Kq, gain_kv=0.0, gain_kd=0.0, gain_km=km, gain_A0=A0)),
    ObserverInfo("Est. pvm", 'b', (5,(10,3)), ComplementaryINS(gain_kp=kp, gain_kc=kc, gain_Kq = Kq, gain_kv=kv, gain_kd=kd, gain_km=km, gain_A0=A0)),
    # ObserverInfo("Est. v", 'y', ':', ComplementaryINS(gain_kp=0.0, gain_kc=0.0, gain_Kq = Kq, gain_kv=kv, gain_kd=kd, gain_km=0.0)),
    # ObserverInfo("Est. vm", 'c', ':', ComplementaryINS(gain_kp=0.0, gain_kc=0.0, gain_Kq = Kq, gain_kv=kv, gain_kd=kd, gain_km=km)),
]


circle_frequency = 0.5
def velocity_gen(t, X):
    omega = np.reshape([0, 0, circle_frequency], (3, 1))
    g = 9.81 * np.reshape([0, 0, 1], (3, 1))
    a =  - (circle_frequency**2)*X[0:3,0:3].T @ X[0:3,4:5] - X[0:3,0:3].T @ g

    U = np.block([
        [SO3.skew(omega), a, np.zeros((3, 1))],
        [np.zeros((2, 5))]
    ])
    G = np.block([
        [np.zeros((3, 3)), g, np.zeros((3, 1))],
        [np.zeros((2, 5))]
    ])
    N = np.block([
        [np.zeros((3, 3)), np.zeros((3, 2))],
        [np.zeros((1, 3)), 0, -1],
        [np.zeros((1, 3)), 0, 0]
    ])

    return U, G, N

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


def run_once(observer_list):
    X = get_true_state_from_df(df,0)
    X[:3, 3] = np.zeros(3)
    make_initial_condition = lambda x: x # X @ SE23.exp(0.02*np.random.randn(9,1)).as_matrix()
    initial_condition = make_initial_condition(X)
    if args.extreme_initial:
        initial_condition = X @ SE23.exp(np.reshape((0.99*np.pi,0,0,0,0,0,0,0,0), (9,1))).as_matrix()
        initial_condition[0:3,3] += [2,2,0]
        initial_condition[0:3,4] += [1,1,0]
    print("Initial Condition:")
    print(f"Rotation mat = {X[:3, :3]}")
    print(f"Vel  = {X[:3, 3]}")
    print(f"Pos = {X[:3, 4]}")
    print(f"Full state: ")
    print(initial_condition)
    
    for obs in observer_list:
        obs.obs.set_IC(initial_condition)
        obs.obs.ZHat[0:3,3:5] = initial_condition[0:3,3:5] @ obs.obs.ZHat[3:5,3:5]

    statesTru = []
    for step in tqdm.tqdm(range(max_steps)):
        pos_true = X[0:3,4:5].copy()
        vel_true = X[0:3,3:4].copy()
        mag_true = X[0:3,0:3].T @ m0

        X = get_true_state_from_df(df, step)

        omega = get_from_df(df, "imu_angular_vel",Dimensions.vec3(), step).to_numpy().reshape(3, 1)
        imu_angular_vel_skewed = SO3.skew(omega)
        imu_lin_accel = get_from_df(df,"imu_linear_acc",Dimensions.vec3(), step).to_numpy().reshape(3, 1)        
        
        gyr, acc = SO3.vex(imu_angular_vel_skewed), imu_lin_accel

        for obs_data in observer_list:
            obs_data.obs.GPS_update(pos_true, vel_true)
            obs_data.obs.compass_update(mag_true, m0)
            obs_data.obs.integrate_dynamics(gyr, acc, dt)

            obs_data.states_est.append(obs_data.obs.XHat.copy())
            obs_data.states_aux.append(obs_data.obs.ZHat.copy())

        # Gather information
        statesTru.append(X.copy())
    return statesTru, observer_list

times = [(dt / sim_speed_multiplier) * step for step in range(0, max_steps,sim_speed_multiplier)]
statesTru, observer_list = run_once(observer_list)


x = df['odom_pose_x'].to_numpy()[::sim_speed_multiplier]
y = df['odom_pose_y'].to_numpy()[::sim_speed_multiplier]

fig, ax = plt.subplots()
sc = ax.scatter(x,y,
                c=times, s=5)
plt.colorbar(sc, label='time (s)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('XY Trajectory Colored by Time')
ax.grid(True)
ax.axis('equal')

# Plot the error statistics
fig, ax = plt.subplots(4, 1, layout='constrained')
fig.set_figwidth(3.5*figsize_factor)
fig.set_figheight(figheight*figsize_factor)
for obs in observer_list:
    states_est = obs.states_est
    states_aux = obs.states_aux

    att_error = np.array([np.linalg.norm(SO3.log(SO3.from_matrix(XTru[0:3,0:3] @ XEst[0:3,0:3].T)))
                        * 180.0 / np.pi for XTru, XEst in zip(statesTru, states_est)])
    ax[0].plot(times, att_error, linestyle=obs.ls, color=obs.lc, label=obs.name)

    vel_error = np.array([np.linalg.norm(XTru[0:3,3:4] - XEst[0:3,3:4]) for XTru, XEst in zip(statesTru, states_est)])
    ax[1].plot(times, vel_error, linestyle=obs.ls, color=obs.lc, label=obs.name)

    pos_error = np.array([np.linalg.norm(XTru[0:3,4:5] - XEst[0:3,4:5]) for XTru, XEst in zip(statesTru, states_est)])
    ax[2].plot(times, pos_error, linestyle=obs.ls, color=obs.lc, label=obs.name)


    EBar = [np.linalg.inv(ZHat)@(XTru@np.linalg.inv(XEst))@ZHat
            for XTru, XEst, ZHat in zip(statesTru, states_est, states_aux)]
    R_EBar = [EBar_i[0:3,0:3] for EBar_i in EBar] # Attitude
    V_EBar = [EBar_i[0:3,3:5] for EBar_i in EBar] # Position and Velocity
    lyapunov = np.array([
        np.trace(np.eye(3) - R_EBar_i) + np.linalg.norm(x_EBar_i)**2
        for R_EBar_i, x_EBar_i in zip(R_EBar, V_EBar)
    ])
    ax[3].plot(np.reshape(times, [1, -1]).T, lyapunov.T, linestyle=obs.ls, color=obs.lc, label=obs.name)

# Format the axes and figure

ax[0].set_ylabel("Attitude\nError (deg)")
ax[1].set_ylabel("Velocity\nError (m/s)")
ax[2].set_ylabel("Position\nError (m)")
ax[2].legend(ncol=2)

for i in range(3):
    # ax[i].set_ylim([0, None])
    ax[i].set_xlim([times[0], times[-1]])
    ax[i].set_xticklabels([])
    ax[i].grid()


ax[3].set_ylabel("Lyapunov\nValue")
ax[3].set_yscale('log')
max_power = np.floor(np.log10(np.max(lyapunov)))
min_power = np.ceil(np.log10(np.min(lyapunov[lyapunov > 1e-15])))
ax[3].set_yticks([10.0**(i) for i in np.linspace(min_power, max_power, 3, dtype=int)])
ax[3].set_ylim([10**(min_power-1), None])
ax[3].grid()

ax[-1].set_xlim([times[0], times[-1]])
ax[-1].set_xlabel("Time (s)")

fig.suptitle("Observer Error Metrics")


# Plot the estimation over time

fig2, ax2 = plt.subplots(3, 3, layout='constrained')
fig2.set_figwidth(7.16*figsize_factor)
fig2.set_figheight(3/4*figheight*figsize_factor)

eul_tru = np.vstack([SO3.from_matrix(XTru[0:3,0:3]).as_euler() for XTru in statesTru]).T
vel_tru = np.hstack([XTru[0:3,3:4] for XTru in statesTru])
pos_tru = np.hstack([XTru[0:3,4:5] for XTru in statesTru])
for i in range(3):
    for obs in observer_list:
        eul_est = np.vstack([SO3.from_matrix(XEst[0:3,0:3]).as_euler() for XEst in obs.states_est]).T
        vel_est = np.hstack([XEst[0:3,3:4] for XEst in obs.states_est])
        pos_est = np.hstack([XEst[0:3,4:5] for XEst in obs.states_est])

        ax2[i, 0].plot(times, eul_est[i, :], linestyle=obs.ls, color=obs.lc, label=obs.name)
        ax2[i, 1].plot(times, vel_est[i, :], linestyle=obs.ls, color=obs.lc, label=obs.name)
        ax2[i, 2].plot(times, pos_est[i, :], linestyle=obs.ls, color=obs.lc, label=obs.name)
        
    ax2[i, 0].plot(times, eul_tru[i, :], 'k', label='True')
    ax2[i, 1].plot(times, vel_tru[i, :], 'k', label='True')
    ax2[i, 2].plot(times, pos_tru[i, :], 'k', label='True')

    ax2[i, 0].set_xlim([times[0], times[-1]])
    ax2[i, 1].set_xlim([times[0], times[-1]])
    ax2[i, 2].set_xlim([times[0], times[-1]])
    ax2[i, 0].grid()
    ax2[i, 1].grid()
    ax2[i, 2].grid()

    if i < 2:
        ax2[i, 0].set_xticklabels([])
        ax2[i, 1].set_xticklabels([])
        ax2[i, 2].set_xticklabels([])
    else:
        ax2[i, 0].set_xlabel("Time (s)")
        ax2[0, 2].legend(loc='upper right',ncol=2)
        ax2[i, 1].set_xlabel("Time (s)")
        ax2[i, 2].set_xlabel("Time (s)")


ax2[0, 0].set_title("Attitude Estimation")
ax2[0, 0].set_ylabel("roll (deg)")
ax2[1, 0].set_ylabel("pitch (deg)")
ax2[2, 0].set_ylabel("yaw (deg)")
ax2[0, 0].set_ylim([-180.0, 180.0])
ax2[1, 0].set_ylim([-180.0/2, 180.0/2])
ax2[2, 0].set_ylim([-180.0, 180.0])
ax2[0, 1].set_title("Velocity Estimation")
ax2[0, 1].set_ylabel("x (m/s)")
ax2[1, 1].set_ylabel("y (m/s)")
ax2[2, 1].set_ylabel("z (m/s)")
ax2[0, 2].set_title("Position Estimation")
ax2[0, 2].set_ylabel("x (m)")
ax2[1, 2].set_ylabel("y (m)")
ax2[2, 2].set_ylabel("z (m)")

# Auxiliary states
fig3, ax3 = plt.subplots(3, 2, layout='constrained')
fig3.set_figwidth(7.16*figsize_factor)
fig3.set_figheight(figheight*figsize_factor)

for i in range(3):
    for obs in observer_list:
        temp = [ZHat[0:3, 3:5] @ np.linalg.inv(ZHat[3:5,3:5]) for ZHat in obs.states_aux]
        vel_aux = np.hstack([V[0:3, 0:1] for V in temp])
        pos_aux = np.hstack([V[0:3, 1:2] for V in temp])
        ax3[i, 0].plot(times, vel_tru[i,:] - vel_aux[i, :], linestyle=obs.ls, color=obs.lc, label=obs.name)
        ax3[i, 1].plot(times, pos_tru[i,:] - pos_aux[i, :], linestyle=obs.ls, color=obs.lc, label=obs.name)

    ax3[i, 0].set_xlim([times[0], times[-1]])
    ax3[i, 1].set_xlim([times[0], times[-1]])
    ax3[i, 0].grid()
    ax3[i, 1].grid()

    if i < 2:
        ax3[i, 0].set_xticklabels([])
        ax3[i, 1].set_xticklabels([])
    else:
        ax3[i, 0].set_xlabel("Time (s)")
        ax3[i, 1].set_xlabel("Time (s)")

ax3[0, 0].set_title("Velocity Excitation $\mu_v$")
ax3[0, 0].set_ylabel("x (m/s)")
ax3[1, 0].set_ylabel("y (m/s)")
ax3[2, 0].set_ylabel("z (m/s)")
ax3[0, 1].set_title("Position Excitation $\mu_p$")
ax3[0, 1].set_ylabel("x (m)")
ax3[1, 1].set_ylabel("y (m)")
ax3[2, 1].set_ylabel("z (m)")
plt.show()
# Animate trajectories
if args.animate:
    # from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    from pylie import plotting


    fig_ani = plt.figure(layout='constrained')
    fig_ani.set_figwidth(3.5*figsize_factor)
    fig_ani.set_figheight(figheight*figsize_factor)
    ax = fig_ani.add_subplot(111, projection='3d')

    line_length = 20.
    pos_tru = np.stack([X[0:3, 4] for X in statesTru])
    pos_all = pos_tru
    poses_tru = [SE3(SO3.from_matrix(X[0:3,0:3]), R3(X[0:3,4])) for X in statesTru]
    frame_artist_tru = plotting.plotFrame(poses_tru[0], style='-', colors='k', length=line_length, ax=ax)
    for obs in observer_list:
        obs.pos_est = np.stack([X[0:3, 4] for X in obs.states_est])
        pos_all = np.vstack((pos_all, obs.pos_est))
        obs.poses_est = [SE3(SO3.from_matrix(X[0:3,0:3]), R3(X[0:3,4])) for X in obs.states_est]
        obs.frame_artist_est = plotting.plotFrame(obs.poses_est[0], obs.ls, colors=obs.lc, length=line_length, ax=ax)

    ax.set_xlim(min(pos_all[:,0])-line_length, max(pos_all[:,0])+line_length)
    ax.set_ylim(min(pos_all[:,1])-line_length, max(pos_all[:,1])+line_length)
    ax.set_zlim(min(pos_all[:,2])-line_length, max(pos_all[:,2])+line_length)

    ax.set_aspect('equal')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    def update_animation(frame):
        # global ax
        if frame < len(poses_tru):
            frame_artist_tru.set_pose_data(poses_tru[min(frame,len(poses_tru)-1)])
            for obs in observer_list:
                obs.frame_artist_est.set_pose_data(obs.poses_est[min(frame,len(obs.poses_est)-1)])
    
    ani = FuncAnimation(fig_ani, update_animation, frames=len(poses_tru)+500, interval=20)
    ax.view_init(30, 70)
    print("Saving animation...")
    ani.save('INS_animation.mp4', fps=50)


if args.extreme_initial:
    fig.savefig("INS_observer_error_extreme.pdf", bbox_inches = 'tight', pad_inches = 0.02)
    fig2.savefig("INS_estimation_extreme.pdf", bbox_inches = 'tight', pad_inches = 0.02)
else:
    fig.savefig("INS_observer_error_standard.pdf", bbox_inches = 'tight', pad_inches = 0.02)
    fig2.savefig("INS_estimation_standard.pdf", bbox_inches = 'tight', pad_inches = 0.02)


plt.show()
