import numpy as np
import cv2
import normalized_velocity_control as nvc
import stretch_body.robot as rb
import time
import yaml
from yaml.loader import SafeLoader
from scipy.spatial.transform import Rotation
from hello_helpers import hello_misc as hm
import argparse
import zmq
import loop_timer as lt
import forcesight_networking as fn
from pprint import pprint

import finger_force as ff

    
####################################
# Miscellaneous Parameters

motion_on = True

stop_if_goal_not_detected_this_many_frames = 10 #4 #1
stop_if_fingers_not_detected_this_many_frames = 10 #4 #1

# Defines a deadzone for mobile base rotation, since low values can
# lead to no motion and noises on some surfaces like carpets.
min_base_speed = 0.01 #0.05

# Force control parameters
grip_force_dead_zone = 3.0
grip_force_gain = 0.5 #1.0

lift_force_dead_zone = 1.0
lift_force_gain = 0.01 #0.015

# Success criteria
grip_force_error_threshold = 3.0 # 3.0
lift_force_error_threshold = 3.0 # 2.0
grip_center_error_threshold = 0.02 # 0.02
grip_width_error_threshold = 0.02 # 0.02


####################################
## Control Loop Frequency Regulation

# Target control loop rate used when receiving task-relevant
# information from an external process, instead of directly acquiring
# images from the D405. Directly acquiring images uses a blocking call
# and the D405 provides images at a consistent rate, which regulates
# the control loop. Receiving task-relevant information from an
# external process uses nonblocking polling of communications that can
# be highly variable due to communication and computation timing. The
# polling timeout is set automatically in an attempt to approximate
# this target control loop frequency.
target_control_loop_rate_hz = 15
# Proportional gain used to regulate the polling timeout to achieve the target frequency
timeout_proportional_gain = 0.1
# How much history to consider when regulating the polling timeout.
seconds_of_timing_history = 1

####################################
## Gains for Reach Behavior

overall_visual_servoing_velocity_scale = 0.2 #0.3 #0.4 #0.2 #0.5 #0.8 #0.5 #0.1 #1.0

max_distance_to_grip_center_goal = 0.4

joint_visual_servoing_velocity_scale = {
    'base_forward' : 15.0,
    'lift_up' : 20.0,
    'arm_out' : 20.0,
    'wrist_yaw_counterclockwise' : 2.0,
    'wrist_pitch_up' : 6.0,
    'wrist_roll_counterclockwise': 1.0,
    'gripper_open' : 5.0
}

####################################
## Initial Pose

joint_state_center = {
    'lift_pos' : 0.65,
    'arm_pos': 0.01,
    'wrist_yaw_pos': 0.0,
    'wrist_pitch_pos': -0.3, #-0.35, #-0.3, #-0.25, #-0.2
    'wrist_roll_pos': 0.0,
    'gripper_pos': 10.46
}

####################################
## Gains for Achieving Initial Pose

recenter_velocity_scale = {
    'lift_up': 4.0,
    'arm_out': 4.0,
    'wrist_yaw_counterclockwise': 1.5,
    'wrist_pitch_up': 1.5,
    'wrist_roll_counterclockwise': 1.5,
    'gripper_open': 0.5
}

####################################
## Allowed Range of Motion

min_joint_state = {
    'base_odom_x' : -0.2,
    'lift_pos': 0.1,
    'arm_pos': 0.01, #0.03
    'wrist_yaw_pos': -0.20, #-0.25
    'wrist_pitch_pos': -1.2,
    'wrist_roll_pos': -0.1,
    'gripper_pos' : -3.6 #3.5 #4.0 #3.0 
    }

max_joint_state = {
    'base_odom_x' : 0.2,
    'lift_pos': 1.05, #
    'arm_pos': 0.45,
    'wrist_yaw_pos': 1.0, #0.5
    'wrist_pitch_pos': 0.2, #-0.4
    'wrist_roll_pos': 0.1,
    'gripper_pos': 10.4
    }


####################################
## Zero Velocity Command

zero_vel = {
    'base_forward': 0.0,
    'lift_up': 0.0,
    'arm_out': 0.0,
    'wrist_yaw_counterclockwise': 0.0,
    'wrist_pitch_up': 0.0,
    'wrist_roll_counterclockwise': 0.0,
    'gripper_open': 0.0
}

####################################
## Translate Between Keys

pos_to_vel_cmd = {
    'base_odom_x' : 'base_forward', 
    'lift_pos':'lift_up', 
    'arm_pos':'arm_out',
    'wrist_yaw_pos':'wrist_yaw_counterclockwise',
    'wrist_pitch_pos':'wrist_pitch_up',
    'wrist_roll_pos':'wrist_roll_counterclockwise',
    'gripper_pos':'gripper_open'
}

vel_cmd_to_pos = { v:k for (k,v) in pos_to_vel_cmd.items() }

####################################

class RegulatePollTimeout:
    def __init__(self, target_control_loop_rate_hz, seconds_of_timing_history, timeout_proportional_gain, debug_on=False):

        self.debug_on = debug_on
        self.target_control_loop_rate_hz = target_control_loop_rate_hz
        self.seconds_of_timing_history = seconds_of_timing_history
        self.timeout_proportional_gain = timeout_proportional_gain

        self.target_control_loop_period_ms = 1000.0 * (1.0/self.target_control_loop_rate_hz)
        self.initial_timeout_for_socket_poll_ms = self.target_control_loop_period_ms
        self.timeout_for_socket_poll_ms = self.target_control_loop_period_ms
        
        self.recent_polling_durations_max_length = self.seconds_of_timing_history * int(round(self.target_control_loop_rate_hz))
        self.recent_non_polling_durations_max_length = self.seconds_of_timing_history * int(round(self.target_control_loop_rate_hz))
        
        self.time_before_socket_poll = None
        self.prev_time_before_socket_poll = None
        self.time_after_socket_poll = None
        self.prev_time_after_socket_poll = None
        
        self.recent_polling_durations = []
        self.recent_non_polling_durations = []
        
    def run_after_polling(self):
        self.prev_time_after_socket_poll = self.time_after_socket_poll
        self.time_after_socket_poll = time.time()

    def get_poll_timeout(self): 
        # When obtaining task-relevant information via a
        # socket, the required processing should be low. Only
        # robot communication is likely to take significant
        # time. Consequently, the timeout for polling is
        # expected to represent a majority of the period for
        # the control loop. This attempts to select a polling
        # timeout that will result in the control loop being
        # close to the target frequency. Ultimately,
        # performance will depend on the rate at which
        # task-relevant information is received, but motor
        # control behavior will be more consistent.
        
        self.prev_time_before_socket_poll = self.time_before_socket_poll
        self.time_before_socket_poll = time.time()
        
        mean_polling_duration_ms = None
        mean_non_polling_duration_ms = None

        if self.debug_on: 
            print('--------------------------------------------------')
            print('RegulatePollTimeout: get_poll_timeout()')
            print('self.initial_timeout_for_socket_poll_ms =', self.initial_timeout_for_socket_poll_ms)
        
        if (self.time_after_socket_poll is not None) and (self.prev_time_before_socket_poll is not None):
            self.recent_polling_durations.append(self.time_after_socket_poll - self.prev_time_before_socket_poll)
            if len(self.recent_polling_durations) > self.recent_polling_durations_max_length:
                self.recent_polling_durations.pop(0)
            mean_polling_duration_ms = 1000.0 * np.mean(np.array(self.recent_polling_durations))
            if self.debug_on: 
                print('mean_polling_duration_ms =', mean_polling_duration_ms)

        if (self.time_after_socket_poll is not None) and (self.time_before_socket_poll is not None):
            self.recent_non_polling_durations.append(self.time_before_socket_poll - self.time_after_socket_poll)                   
            if len(self.recent_non_polling_durations) > self.recent_non_polling_durations_max_length:
                self.recent_non_polling_durations.pop(0)
            mean_non_polling_duration_ms = 1000.0 * np.mean(np.array(self.recent_non_polling_durations))
            if self.debug_on: 
                print('mean_non_polling_duration_ms =', mean_non_polling_duration_ms)

        if (mean_polling_duration_ms is not None) and (mean_non_polling_duration_ms is not None):
            mean_full_duration_ms = mean_polling_duration_ms + mean_non_polling_duration_ms
            full_duration_error_ms = self.target_control_loop_period_ms - mean_full_duration_ms
            self.timeout_for_socket_poll_ms = self.timeout_for_socket_poll_ms + (self.timeout_proportional_gain * full_duration_error_ms)

            if self.debug_on: 
                print('self.target_control_loop_perios_ms =', self.target_control_loop_period_ms)
                print('mean_full_duration_ms =', mean_full_duration_ms)
                print('full_duration_error_ms =', full_duration_error_ms)
                print('self.timeout_proportional_gain =', self.timeout_proportional_gain)
                print('self.timeout_for_socket_poll_ms =', self.timeout_for_socket_poll_ms)

        timeout_for_socket_poll_ms_int = int(round(self.timeout_for_socket_poll_ms))
        if timeout_for_socket_poll_ms_int <= 0:
            timeout_for_socket_poll_ms_int = 1

        if self.debug_on: 
            print('timeout_for_socket_poll_ms_int =', timeout_for_socket_poll_ms_int)
            print('--------------------------------------------------')
        return timeout_for_socket_poll_ms_int

        

def recenter_robot(controller):
    centered = False
    overall_velocity_scale = 1.0
    wait_time = 0.05
    low_enough_total_abs_error = 0.15
  
    while not centered:
        joint_state = controller.get_joint_state()
        #print('joint_state =', joint_state)
        joint_errors = {k: (v - joint_state[k]) for (k,v) in joint_state_center.items()}
        #print('joint_errors =', joint_errors)
        total_abs_error = sum([abs(v) for v in joint_errors.values()])
        
        #print('total_abs_error = ', total_abs_error)
        if total_abs_error > low_enough_total_abs_error:
            joint_velocity = {k: overall_velocity_scale * v for (k,v) in joint_errors.items()}
            joint_velocity_cmd = {pos_to_vel_cmd[k]: v for (k,v) in joint_velocity.items()}
            cmd = {k: recenter_velocity_scale[k] * v for (k,v) in joint_velocity_cmd.items()}
            #print('joint_velocity_cmd =', joint_velocity_cmd)
            cmd = { k: ( 0.0 if ((v < 0.0) and (joint_state[vel_cmd_to_pos[k]] < min_joint_state[vel_cmd_to_pos[k]])) else v ) for (k,v) in cmd.items()}
            cmd = { k: ( 0.0 if ((v > 0.0) and (joint_state[vel_cmd_to_pos[k]] > max_joint_state[vel_cmd_to_pos[k]])) else v ) for (k,v) in cmd.items()}
            controller.set_command(cmd)
            time.sleep(wait_time)
        else:
            centered = True
            controller.set_command(nvc.zero_vel)
    controller.reset_base_odometry()

    
    
def main(use_remote_computer):

    np.set_printoptions(precision=3, linewidth=100, suppress=True)

    finger_force = ff.FingerForce()
    finger_force.prepare_lookup_table()
    
    forcesight_context = zmq.Context()
    forcesight_socket = forcesight_context.socket(zmq.SUB)
    forcesight_socket.setsockopt(zmq.SUBSCRIBE, b'')
    forcesight_socket.setsockopt(zmq.SNDHWM, 1)
    forcesight_socket.setsockopt(zmq.RCVHWM, 1)
    forcesight_socket.setsockopt(zmq.CONFLATE, 1)
    if use_remote_computer:
        forcesight_address = 'tcp://' + fn.remote_computer_ip + ':' + str(fn.forcesight_port)
    else:
        forcesight_address = 'tcp://' + '127.0.0.1' + ':' + str(fn.forcesight_port)
    forcesight_socket.connect(forcesight_address)
    
    regulate_socket_poll = RegulatePollTimeout(target_control_loop_rate_hz,
                                               seconds_of_timing_history,
                                               timeout_proportional_gain,
                                               debug_on=False)

    
    action_status_context = zmq.Context()
    action_status_socket = action_status_context.socket(zmq.PUB)
    action_status_address = 'tcp://*:' + str(fn.action_status_port)
    action_status_socket.setsockopt(zmq.SNDHWM, 1)
    action_status_socket.setsockopt(zmq.RCVHWM, 1)
    action_status_socket.bind(action_status_address)

    try:
        first_frame = True

        robot = rb.Robot()
        robot.startup()

        pan = np.pi/2.0
        tilt = -np.pi/2.0

        robot.head.move_to('head_pan', pan)
        robot.head.move_to('head_tilt', tilt)
        robot.push_command()
        time.sleep(1.0)

        controller = nvc.NormalizedVelocityControl(robot)
        v = 0.05

        recenter_robot(controller)

        # model zero pitch force (compensate for gravity)
        i = 0.0
        total_pitch_eff = 0.0
        for n in range(20):
            joint_state = controller.get_joint_state()
            total_pitch_eff = total_pitch_eff + joint_state['wrist_pitch_eff']
            i = i + 1.0
            time.sleep(0.1)
        non_contact_pitch_eff = total_pitch_eff / i
            
        frames_since_goal_detected = 0
        frames_since_fingers_detected = 0
            
        loop_timer = lt.LoopTimer()

        fingertips = {}
        
        while True:
            loop_timer.start_of_iteration()

            prompt = None
            
            grip_center = None
            grip_center_goal = None
            grip_center_error = None
            grip_center_error_magnitude = None

            grip_width = None
            grip_width_goal = None
            grip_width_error = None
            grip_width_error_magnitude = None

            gripper_yaw_goal = None

            fingertip_left_pos = None       
            fingertip_right_pos = None

            grip_force = None
            grip_force_goal = None
            grip_force_error = None

            lift_force = None
            lift_force_goal = None
            lift_force_error = None

            grip_force_success = False
            lift_force_success = False
            grip_width_success = False
            grip_center_success = False
            
            timeout_for_socket_poll_int = regulate_socket_poll.get_poll_timeout()
            #print('timeout_for_socket_poll_int =', timeout_for_socket_poll_int)
            poll_results = forcesight_socket.poll(timeout=timeout_for_socket_poll_int,
                                            flags=zmq.POLLIN)
            if poll_results == zmq.POLLIN:
                forcesight_results = forcesight_socket.recv_pyobj()
                #print('forcesight_results =', forcesight_results)
                fingertips = forcesight_results.get('fingertips', None)
                forcesight = forcesight_results.get('forcesight')
                if forcesight is not None:
                    prompt = forcesight['prompt']
                    confidence = forcesight['confidence']
                    confidence_threshold = 0.002
                    if confidence is not None: 
                        if confidence > confidence_threshold:
                            grip_center_goal = forcesight['grip_center']['xyz_m']
                            grip_width_goal = forcesight['grip_width_m']
                            gripper_yaw_goal = forcesight['gripper_yaw_rad']
                            grip_force_goal = forcesight['grip_force_n']
                            lift_force_goal = forcesight['applied_force_camera_n'][1]


            if prompt is not None:
                if "lift" in prompt:
                    robot.lift.move_by(0.2)
                    robot.push_command()
                    time.sleep(2.0)
                    
                    print('*********** TOTAL SUCCESS!!!!!!!!!! ************')
                    action_status = {
                        'prompt': prompt,
                        'successful': True
                        }
                    print('sending action status =')
                    print(action_status)
                    action_status_socket.send_pyobj(action_status)
            
            regulate_socket_poll.run_after_polling()

            fingertip_left_pose = None
            fingertip_right_pose = None
            f = fingertips.get('left', None)
            if f is not None:
                fingertip_left_pos = f['pos']
            f = fingertips.get('right', None)
            if f is not None:
                fingertip_right_pos = f['pos']
            
            if (fingertip_left_pos is not None) and (fingertip_right_pos is not None):
                grip_center = (fingertip_left_pos + fingertip_right_pos)/2.0
                grip_width = np.linalg.norm(fingertip_left_pos - fingertip_right_pos)

            joint_state = controller.get_joint_state()

            grip_pos = joint_state['gripper_pos']
            grip_eff = joint_state['gripper_eff']
            pitch_eff = joint_state['wrist_pitch_eff']
            yaw_eff = joint_state['wrist_yaw_eff']
            arm_eff = joint_state['arm_eff']
            lift_eff = joint_state['lift_eff']
            finger_forces = None
            if fingertips is not None:
                finger_forces = finger_force.forcesight_forces_using_efforts(grip_pos, fingertips, pitch_eff, yaw_eff, arm_eff, non_contact_pitch_eff)
                grip_force = finger_forces['grip_force']
                if grip_force is not None: 
                    if grip_force < 0.0:
                        grip_force = 0.0
                if (finger_forces is not None):
                    if finger_forces['applied_force'] is not None: 
                        lift_force = finger_forces['applied_force'][1]
                if (grip_force_goal is not None) and (grip_force is not None):
                    grip_force_error = grip_force_goal - grip_force
                if (lift_force_goal is not None) and (lift_force is not None):
                    lift_force_error = lift_force_goal - lift_force
                    
            if grip_center_goal is not None:
                frames_since_goal_detected = 0
            else:
                frames_since_goal_detected = frames_since_goal_detected + 1

            if grip_center is not None:
                frames_since_fingers_detected = 0
            else: 
                frames_since_fingers_detected = frames_since_fingers_detected + 1

            if (grip_center is not None) and (grip_center_goal is not None):            

                grip_center_error = grip_center_goal - grip_center
                grip_center_error_magnitude = np.linalg.norm(grip_center_error)

                if False: 
                    print()
                    print('grip_center =', grip_center)
                    print('grip_center_goal =', grip_center_goal)
                    print('grip_center_error_magnitude = {:.3f}'.format(grip_center_error_magnitude))

            if (grip_width is not None) and (grip_width_goal is not None):            

                grip_width_error = grip_width_goal - grip_width
                grip_width_error_magnitude = abs(grip_width_error)
                if False: 
                    print()
                    print('grip_width = {:.3f}'.format(grip_width))
                    print('grip_width_goal = {:.3f}'.format(grip_width_goal))
                    print('grip_width_error = {:.3f}'.format(grip_width_error))

            if (gripper_yaw_goal is not None):
                gripper_yaw = joint_state['wrist_yaw_pos']
                gripper_yaw_error = gripper_yaw_goal - gripper_yaw
                if False: 
                    print('gripper_yaw = {:.3f}'.format(gripper_yaw))
                    print('gripper_yaw_goal = {:.3f}'.format(gripper_yaw_goal))
                    print('gripper_yaw_error = {:.3f}'.format(gripper_yaw_error))


            if grip_force_error is not None:
                print('abs(grip_force_error) =', abs(grip_force_error))
                grip_force_success = abs(grip_force_error) < grip_force_error_threshold
                if grip_force_success:
                    print('GRIP FORCE SUCCESS')

            if lift_force_error is not None:
                print('abs(lift_force_error) =', abs(lift_force_error))
                lift_force_success = abs(lift_force_error) < lift_force_error_threshold
                if lift_force_success:
                    print('LIFT FORCE SUCCESS')

            if grip_center_error is not None:
                print('np.linalg.norm(grip_center_error) =', np.linalg.norm(grip_center_error))
                grip_center_success = np.linalg.norm(grip_center_error) < grip_center_error_threshold
                if grip_center_success:
                    print('GRIP CENTER SUCCESS')

            if grip_width_error is not None:
                print('abs(grip_width_error) =', abs(grip_width_error))
                grip_width_success = abs(grip_width_error) < grip_width_error_threshold
                if grip_width_success:
                    print('GRIP WIDTH SUCCESS')

            if grip_force_success and lift_force_success and grip_center_success and grip_width_success:
                print('*********** TOTAL SUCCESS!!!!!!!!!! ************')
                action_status = {
                    'prompt': prompt,
                    'successful': True
                    }
                print('sending action status =')
                print(action_status)
                action_status_socket.send_pyobj(action_status)
            
                    
            if (grip_center_error is not None) or (grip_width_error is not None):
                cmd = {}

                if gripper_yaw_goal is not None: 
                      yaw_velocity = gripper_yaw_goal - joint_state['wrist_yaw_pos']
                else:
                      yaw_velocity = 0.0
                pitch_velocity = joint_state_center['wrist_pitch_pos'] - joint_state['wrist_pitch_pos']
                roll_velocity =  0.0 - joint_state['wrist_roll_pos']

                cmd['wrist_yaw_counterclockwise'] = yaw_velocity
                cmd['wrist_pitch_up'] = pitch_velocity
                cmd['wrist_roll_counterclockwise'] = roll_velocity

                if (grip_center_error is not None) and (grip_center_error_magnitude < max_distance_to_grip_center_goal): 
                    x_error, y_error, z_error = grip_center_error

                    # Transform camera frame errors into errors for the Cartesian joints
                    yaw = joint_state['wrist_yaw_pos']
                    pitch = -joint_state['wrist_pitch_pos']
                    roll = -joint_state['wrist_roll_pos']
                    r = Rotation.from_euler('yxz', [yaw, pitch, roll]).as_matrix()
                    rotated_lift = np.matmul(r, np.array([0.0, -1.0, 0.0]))
                    rotated_arm = np.matmul(r, np.array([0.0, 0.0, 1.0]))
                    rotated_base = np.matmul(r, np.array([-1.0, 0.0, 0.0]))

                    lift_velocity = np.dot(rotated_lift, grip_center_error)

                    print()
                    print(prompt)
                    if (lift_force_goal is not None) and (lift_force is not None):
                        lift_force_error = lift_force_goal - lift_force
                        if abs(lift_force_error) > lift_force_dead_zone:
                            gain = lift_force_gain/lift_force_dead_zone
                            delta = gain * lift_force_error
                            lift_velocity = lift_velocity + delta
                            print('bang! lift force control on with delta =', delta)
                            
                            #delta = np.sign(lift_force_error) * lift_force_gain
                            #print('bang! lift force control on with delta =', delta)
                            #lift_velocity = lift_velocity + delta

                        print('lift_force_goal =', lift_force_goal)
                        print('lift_force =', lift_force)
                        print('lift_velocity =', lift_velocity)
                        
                    arm_velocity = np.dot(rotated_arm, grip_center_error)

                    #base_rotational_velocity = np.dot(rotated_base, grip_center_error) / (joint_state['arm_pos'] + max_gripper_length)
                    #base_rotational_velocity = np.dot(rotated_base, grip_center_error)
                    #print('base_rotational_velocity =', base_rotational_velocity)
                    #if abs(base_rotational_velocity) < min_base_speed:
                    #    base_rotational_velocity = 0.0
                    
                    base_translational_velocity = np.dot(rotated_base, grip_center_error)
                    #print('base_translational_velocity =', base_translational_velocity)
                    if abs(base_translational_velocity) < min_base_speed:
                        base_translational_velocity = 0.0
                    #print('base_translational_velocity =', base_translational_velocity)
                    #print('base_odom_x =', joint_state['base_odom_x'])

                    cmd['lift_up'] = lift_velocity
                    cmd['arm_out'] = arm_velocity
                    cmd['base_forward'] = base_translational_velocity
                     
                print()
                if (grip_width_error is not None) and (grip_force_error is not None):
                    grip_force_error = grip_force_goal - grip_force
                    if abs(grip_force_error) > grip_force_dead_zone:
                        gain = grip_force_gain/grip_force_dead_zone
                        delta = -gain * grip_force_error
                        gripper_velocity = (grip_width_goal - grip_width) + delta
                        print('bang! grip force control on with delta =', delta)
                        
                        #delta = np.sign(grip_force_error) * grip_force_gain
                        #print('bang! grip force control on with delta =', delta)
                        #gripper_velocity = (grip_width_goal - grip_width) - delta
                    else:
                        gripper_velocity = grip_width_goal - grip_width
                        #grip_force_weight = 0.3 #0.25 #0.18

                    
                    print('grip_width_goal =', grip_width_goal)
                    print('grip_width =', grip_width)
                    print()
                    print('grip_force_goal =', grip_force_goal)
                    print('grip_force =', grip_force)
                    print()
                    print('gripper_velocity =', gripper_velocity)

                    
                    cmd['gripper_open'] = gripper_velocity
                    #print('gripper_velocity =', gripper_velocity)
                
                cmd = {k: overall_visual_servoing_velocity_scale * v for (k,v) in cmd.items()}
                cmd = {k: joint_visual_servoing_velocity_scale[k] * v for (k,v) in cmd.items()}
                
                if motion_on:
                    cmd = { k: ( 0.0 if ((v < 0.0) and (joint_state[vel_cmd_to_pos[k]] < min_joint_state[vel_cmd_to_pos[k]])) else v ) for (k,v) in cmd.items()}
                    cmd = { k: ( 0.0 if ((v > 0.0) and (joint_state[vel_cmd_to_pos[k]] > max_joint_state[vel_cmd_to_pos[k]])) else v ) for (k,v) in cmd.items()}

                    controller.set_command(cmd)

            else:
                joint_state = controller.get_joint_state()
                stop_joints = zero_vel.copy()

                if frames_since_goal_detected >= stop_if_goal_not_detected_this_many_frames:
                    cmd = stop_joints
                elif frames_since_fingers_detected >= stop_if_fingers_not_detected_this_many_frames:
                    cmd = stop_joints
                else:
                    # Stop at Boundaries
                    cmd = { k:v for (k,v) in stop_joints.items() if (joint_state[vel_cmd_to_pos[k]] < min_joint_state[vel_cmd_to_pos[k]]) }
                    cmd = { k:v for (k,v) in stop_joints.items() if (joint_state[vel_cmd_to_pos[k]] > max_joint_state[vel_cmd_to_pos[k]]) }

                if cmd:
                    cmd = { k: ( 0.0 if ((v < 0.0) and (joint_state[vel_cmd_to_pos[k]] < min_joint_state[vel_cmd_to_pos[k]])) else v ) for (k,v) in cmd.items()}
                    cmd = { k: ( 0.0 if ((v > 0.0) and (joint_state[vel_cmd_to_pos[k]] > max_joint_state[vel_cmd_to_pos[k]])) else v ) for (k,v) in cmd.items()}
                    controller.set_command(cmd)
            
            cv2.waitKey(1)

            loop_timer.end_of_iteration()
            #loop_timer.pretty_print()
    finally:
        controller.stop()
        pipeline.stop()




if __name__ == '__main__':

    print()
    print('********************************************************************************')
    print('IMPORTANT: This code uses an unmodified deep model trained with a significantly different robot as part of an academic research project.  When allowing the robot to move based on ForceSight you must be ready to push the run-stop button and terminate the code. The robot will take actions that put itself, its surroundings, and people at risk! Be careful! USE AT YOUR OWN RISK!')
    print('********************************************************************************')
    print()
    
    parser = argparse.ArgumentParser(
        prog='Stretch ForceSight Servoing',
        description='This is a Cartesian visual servoing controller for ForceSight.'
    )

    parser.add_argument('-r', '--remote', action='store_true', help = 'Use this argument when allowing a remote computer to send task-relevant information for visual servoing, such as 3D positions for the fingertips and target objects. Prior to using this option, configure the network with the file forcesight_networking.py.')

    args = parser.parse_args()
    use_remote_computer = args.remote
    main(use_remote_computer)
    
