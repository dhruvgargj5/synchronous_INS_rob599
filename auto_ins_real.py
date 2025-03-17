from ins_observer import ComplementaryINS, calcposNED

from pymavlink import mavutil, mavextra
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interpn

try:
    from progressbar import ProgressBar
except ImportError:
    ProgressBar = lambda x : x

import matplotlib.pyplot as plt

plt.rc('lines', linewidth=1.0)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
CINS_label = 'Ours'

message_types = ['XKF1', 'SIM', 'RFRH', 'RISI',  'RGPI', 'RGPJ', 'RBRI', 'RMGI', 'AHR2']

lc_dict={
    "EKF3":'blue',
    CINS_label:'red',
    "GNSS":'black',
    "AHRS":'green',
    "SIM":'black',
}

ls_dict={
    "EKF3":'-',
    CINS_label:'--',
    "GNSS":'',
    "AHRS":'-.',
    "SIM":':',
}
ms_dict={
    "EKF3":'',
    CINS_label:'',
    "GNSS":'x',
    "AHRS":'',
    "SIM":'',
}

def observe_log(fname):
    print("Processing {}".format(fname))
    mlog = mavutil.mavlink_connection(fname)

    use_mag_flag = True
    observer = ComplementaryINS(
        gain_kp = 1.0,
        gain_kv = 1.0,
        gain_kc = 0.01,
        gain_kd = 0.001,
        gain_km = 2.0e-6*use_mag_flag,
        gain_Kq = np.diag((0.1, 0.02)),
        gain_A0 = np.diag((1.0, 1.0)),
    )

    EKF3_messages = []
    SIM_messages = []
    AHRS_messages = []
    CINS_results = []
    GPS_results = []
    current_ref_time_stamp = None
    GPS_current_status = None
    GPS_reference_coordinates = None
    MAG_reference_field = None
    GPS_current_NED = np.zeros((3,1))

    imu_message_count = 0
    gps_message_count = 0
    mag_message_count = 0
    
    max_messages = 1e8
    prog_bar = ProgressBar(max_value=mlog.data_len)
    prog_bar_idx = 0
    while True:
        message = mlog.recv_match(type=message_types)
        if message is None or prog_bar_idx > max_messages:
            break
        m_type = message.get_type()
        prog_bar_idx += 1
        prog_bar.update(mlog.offset)

        # Handle the various message types

        # XKF1 is the EKF3 messages.
        # message.C refers to the channel. We only care about the first channel
        if m_type == 'XKF1' and message.C == 0:
            EKF3_messages.append(message)

        if m_type == 'SIM':
            SIM_messages.append(message)
                    
        if m_type == 'RFRH':
            current_ref_time_stamp = message
        
        if m_type == 'RGPI' and message.I == 0:
            GPS_current_status = message

        if m_type == 'RGPJ' and message.I == 0 and GPS_current_status.Stat >= 3:
            if GPS_reference_coordinates is None:
                GPS_reference_coordinates = [message.Lat*1e-7, message.Lon*1e-7, message.Alt*1e-2]
                MAG_reference_field = mavextra.expected_earth_field_lat_lon(GPS_reference_coordinates[0], GPS_reference_coordinates[1])
                MAG_reference_field = np.array([[MAG_reference_field.x],[MAG_reference_field.y],[MAG_reference_field.z]])
                with np.printoptions(precision=2):
                    print(r"Magnetic field reference: {} $\mu$T".format(MAG_reference_field.T*0.1))

            GPS_LLA = [message.Lat*1e-7, message.Lon*1e-7, message.Alt*1e-2]
            GPS_VEL = np.reshape([message.VX, message.VY, message.VZ], (3,1))
            GPS_current_NED = calcposNED(*GPS_LLA, *GPS_reference_coordinates)

            GPS_results.append(
                [current_ref_time_stamp.TimeUS*1.e-6, [None,None,None], GPS_VEL.ravel().tolist(), GPS_current_NED.ravel().tolist()]
                )
            gps_message_count += 1

            observer.GPS_update(GPS_current_NED, GPS_VEL)


        if m_type == 'RMGI' and MAG_reference_field is not None:
            # Magnetometer measurement
            measured_field = np.reshape([message.FX,message.FY,message.FZ],(3,1))
            observer.compass_update(measured_field, MAG_reference_field)
            mag_message_count += 1

        if m_type == 'RBRI' and message.I == 0:
            # Barometer measurement
            pass

        if m_type == 'RISI' and message.I == 0:
            # Integrated IMU measurement
            acc = np.array([[message.DVX],[message.DVY],[message.DVZ]]) / message.DVDT
            gyr = np.array([[message.DAX],[message.DAY],[message.DAZ]]) / message.DVDT

            observer.integrate_dynamics(gyr, acc, message.DVDT)

            CINS_AVP = observer.state_estimates()
            CINS_results.append([current_ref_time_stamp.TimeUS*1.0e-6, *CINS_AVP])

            imu_message_count += 1

        
        if m_type == 'AHR2' and GPS_reference_coordinates is not None:
            AHRS_messages.append(message)

    
    prog_bar.finish()
    
    
    # Convert EKF3 and AHRS messages to results
    print("Converting EKF3 and AHRS messages to results format.")

    EKF3_results = [
        [m.TimeUS*1e-6, [m.Roll, m.Pitch, m.Yaw], [m.VN, m.VE, m.VD], [m.PN, m.PE, m.PD]]
        for m in EKF3_messages
    ]
    AHRS_results = [
        [m.TimeUS*1e-6, [m.Roll, m.Pitch, m.Yaw], [None, None, None], [None, None, None]]
        for m in AHRS_messages
    ]
    SIM_results = [
        [m.TimeUS*1e-6, [m.Roll, m.Pitch, m.Yaw], [None, None, None], calcposNED(m.Lat, m.Lng, m.Alt, *GPS_reference_coordinates)]
        for m in SIM_messages
    ]

    elapsed_time = EKF3_results[-1][0] - EKF3_results[0][0]
    print("IMU Frequency: {} Hz".format(imu_message_count / elapsed_time))
    print("Mag Frequency: {} Hz".format(mag_message_count / elapsed_time))
    print("GPS Frequency: {} Hz".format(gps_message_count / elapsed_time))

    results_dict = {
        "EKF3":EKF3_results,
        CINS_label:CINS_results,
        "GNSS":GPS_results,
        #"AHRS":AHRS_results,
        "SIM":SIM_results
    }
    return results_dict


def plot_results(results_dict, tlims=[None,None]):

    fig_avp, ax_avp = plt.subplots(3, 3, layout='constrained')
    desired_width = 7.16
    fig_avp.set_size_inches(desired_width*fig_size_multiplier, 3/4*figheight*fig_size_multiplier)


    for name, results in results_dict.items():
        if len(results) == 0:
            continue
        stamps = np.hstack([r[0] for r in results])
        if tlims[0] is not None and tlims[1] is not None:
            drange = (stamps >= np.max(stamps[stamps<tlims[0]])) * (stamps <= np.min(stamps[stamps>tlims[1]]))
        else:
            drange = np.ones(stamps.shape, dtype=bool)
        attitude = np.vstack([r[1] for r in results])
        velocity = np.vstack([r[2] for r in results])
        position = np.vstack([r[3] for r in results])
      

        for i in range(3):
            ax_avp[i, 0].plot(stamps[drange], attitude[drange, i], ls_dict[name]+ms_dict[name], label=name, color=lc_dict[name])
            ax_avp[i, 1].plot(stamps[drange], velocity[drange, i], ls_dict[name]+ms_dict[name], label=name, color=lc_dict[name])
            ax_avp[i, 2].plot(stamps[drange], position[drange, i], ls_dict[name]+ms_dict[name], label=name, color=lc_dict[name])

            for j in range(3):
                if tlims[0] is not None and tlims[1] is not None:
                    ax_avp[i, j].set_xlim([tlims[0], tlims[1]])
                else:
                    ax_avp[i, j].set_xlim([stamps[0], stamps[-1]])
                ax_avp[i, j].grid(True)

            if i < 2:
                ax_avp[i, 0].set_xticklabels([])
                ax_avp[i, 1].set_xticklabels([])
                ax_avp[i, 2].set_xticklabels([])
            else:
                ax_avp[i, 0].set_xlabel("Time (s)")
                ax_avp[i, 2].legend(loc='upper right')
                ax_avp[i, 1].set_xlabel("Time (s)")
                ax_avp[i, 2].set_xlabel("Time (s)")


    if ax_avp[2,0].get_ylim()[0] < 0. or ax_avp[2,0].get_ylim()[1] > 360.:
        ax_avp[2,0].set_ylim([0, 360]) # Set yaw axis lims

    ax_avp[0, 0].set_title("Attitude Estimation")
    ax_avp[0, 0].set_ylabel("roll (deg)")
    ax_avp[1, 0].set_ylabel("pitch (deg)")
    ax_avp[2, 0].set_ylabel("yaw (deg)")
    ax_avp[0, 1].set_title("Velocity Estimation")
    ax_avp[0, 1].set_ylabel("N (m/s)")
    ax_avp[1, 1].set_ylabel("E (m/s)")
    ax_avp[2, 1].set_ylabel("D (m/s)")
    ax_avp[0, 2].set_title("Position Estimation")
    ax_avp[0, 2].set_ylabel("N (m)")
    ax_avp[1, 2].set_ylabel("E (m)")
    ax_avp[2, 2].set_ylabel("D (m)")

    return fig_avp


def plot_results_trajectory(results_dict):
    # from mpl_toolkits.mplot3d import Axes3D

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    fig_2d, ax_2d = plt.subplots()

    for name, results in results_dict.items():
        if len(results) == 0:
            continue
        # stamps = np.hstack([r[0] for r in results])
        # attitude = np.vstack([r[1] for r in results])
        # velocity = np.vstack([r[2] for r in results])
        position = np.vstack([r[3] for r in results])
      

        ax_3d.plot(position[:,0], position[:,1], position[:,2], ls_dict[name]+ms_dict[name], label=name, color=lc_dict[name])
        ax_3d.grid(True)
        ax_3d.set_xlabel('N (m)')
        ax_3d.set_ylabel('E (m)')
        ax_3d.set_zlabel('D (m)')

        ax_2d.plot(position[:,0], position[:,1], ls_dict[name]+ms_dict[name], label=name, color=lc_dict[name])
        ax_2d.grid(True)
        ax_2d.set_xlabel('N (m)')
        ax_2d.set_ylabel('E (m)')

    ax_3d.legend()
    ax_2d.legend()


    fig_3d.suptitle("Estimated Trajectories")
    fig_2d.suptitle("Estimated Trajectories (top-down)")

    desired_width = 3.0
    fig_3d.set_size_inches(desired_width*fig_size_multiplier, desired_width*fig_size_multiplier)
    fig_3d.tight_layout()

    ax_2d.axis('equal')
    fig_2d.set_size_inches(desired_width*fig_size_multiplier, desired_width*fig_size_multiplier)
    fig_2d.tight_layout()

    return fig_3d, fig_2d


def plot_compare_results(results_dict, name1, name2):

    fig_err, ax_err = plt.subplots(3, 1)

    att_1 = [ Rotation.from_euler('xyz', r[1], degrees=True).as_matrix() for r in results_dict[name1]]
    vel_1 = np.vstack([r[2] for r in results_dict[name1]])
    pos_1 = np.vstack([r[3] for r in results_dict[name1]])
    stamps_1 = np.hstack([r[0] for r in results_dict[name1]])

    att_2 = [ Rotation.from_euler('xyz', r[1], degrees=True).as_matrix() for r in results_dict[name2]]
    vel_2 = np.vstack([r[2] for r in results_dict[name2]])
    pos_2 = np.vstack([r[3] for r in results_dict[name2]])
    stamps_2 = np.hstack([r[0] for r in results_dict[name2]])

    # Align 2 to 1
    att_2 = interpn((stamps_2,), att_2, stamps_1, bounds_error=False)
    vel_2 = interpn((stamps_2,), vel_2, stamps_1, bounds_error=False)
    pos_2 = interpn((stamps_2,), pos_2, stamps_1, bounds_error=False)

    # Compute errors
    att_err = [180.0/np.pi*np.arccos(0.5*(np.trace(R1.T @ R2)-1.0)) for R1, R2 in zip(att_1,att_2)]
    vel_err = np.linalg.norm(vel_2 - vel_1, axis=1)
    pos_err = np.linalg.norm(pos_2 - pos_1, axis=1)

    ax_err[0].plot(stamps_1, att_err, ls_dict[name2]+ms_dict[name2], color=lc_dict[name2])
    ax_err[1].plot(stamps_1, vel_err, ls_dict[name2]+ms_dict[name2], color=lc_dict[name2])
    ax_err[2].plot(stamps_1, pos_err, ls_dict[name2]+ms_dict[name2], color=lc_dict[name2])

    for k in range(3):
        ax_err[k].set_xlim((0,stamps_1[-1]))
        ax_err[k].set_ylim((0,None))
    ax_err[0].set_ylim([0., 180.]) # Set yaw axis lims

    ax_err[0].set_title("Comparison of {} and {} Estimation".format(name1,name2))
    ax_err[0].set_ylabel("Attitude Diff. (deg)")
    ax_err[1].set_ylabel("Velocity Diff. (m/s)")
    ax_err[2].set_ylabel("Position Diff. (m)")



    desired_width = 3.5
    fig_err.set_size_inches(desired_width*fig_size_multiplier, 3.*fig_size_multiplier)

    fig_err.tight_layout()

    return fig_err



    


if __name__ == '__main__':
    fname = "striver-2023-02-18-f22.bin"
    
    results_dict = observe_log(fname)

    fig_size_multiplier = 1.5
    figheight = 2.5

    limited_results_dict = {k:v for k,v in results_dict.items() if k in ['EKF3', CINS_label]}
    fig_avp = plot_results(limited_results_dict)
    fig_avp_brief = plot_results(results_dict, tlims=[274.,276.])
    fig_3d, fig_2d = plot_results_trajectory(limited_results_dict)

    
    figure_ext = '.pdf'
    fig_avp.savefig("real_INS_results_avp"+figure_ext, bbox_inches = 'tight', pad_inches = 0.02)
    fig_avp_brief.savefig("real_INS_results_avp_brief"+figure_ext, bbox_inches = 'tight', pad_inches = 0.02)
    fig_3d.savefig("real_INS_results_3d"+figure_ext, bbox_inches = 'tight', pad_inches = 0.02)
    fig_2d.savefig("real_INS_results_2d"+figure_ext, bbox_inches = 'tight', pad_inches = 0.02)

    plt.show()
