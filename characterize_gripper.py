
## Code derived from opencv_viewer_example.py
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

import stretch_body.robot as rb
import robot_move as rm
import d405_helpers as dh
import pyrealsense2 as rs
import numpy as np
import cv2
import aruco_detector as ad
import aruco_to_fingertips as af
import yaml
import time
import pprint as pp
from yaml.loader import SafeLoader


def get_gripper_state(robot):
    gripper = robot.end_of_arm.motors['stretch_gripper']
    pos = gripper.status['pos']
    pos_pct = gripper.status['pos_pct']
    effort = gripper.status['effort']
    vel = gripper.status['vel']
    return {'pos':pos, 'pos_pct':pos_pct, 'effort':effort, 'vel':vel}
 
def yaml_fingertips(fingertips):
    # Create version that will result in human readable YAML files
    out = {}
    out['left'] = { (k) : (v.tolist() if 'tolist' in dir(v) else v) for k, v in fingertips['left'].items()}
    out['right'] = { (k) : (v.tolist() if 'tolist' in dir(v) else v) for k, v in fingertips['right'].items()}
    return out
    

try:
    first_frame = True
    
    aruco_to_fingertips = af.ArucoToFingertips(default_height_above_mounting_surface=af.suctioncup_height['cup_bottom'])
    
    marker_info = {}
    with open('aruco_marker_info.yaml') as f:
        marker_info = yaml.load(f, Loader=SafeLoader)
        
    aruco_detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=True, use_apriltag_refinement=True)
    aruco_detector.show_debug_images = True

    
    robot = rb.Robot()
    robot.startup()
    
    robot_move = rm.RobotMove(robot, speed='fast')
    robot_move.print_settings()

    just_touching_fingertip_distance = 0.025 #0.03
    fully_open_fingertip_distance_change = 0.0005 #0.002
    
    gripper_start = 0.0
    gripper_change_per_timestep = 2.0
    max_gripper_change = 300.0
    prev_gripper_command = gripper_start
    
    starting_configuration = {
        'stretch_gripper' : gripper_start
    }
    
    robot_move.to_configuration(starting_configuration, speed='fast')
    robot.push_command()
    time.sleep(2.0)
    
    #robot.wait_command()    

    prev_distance_between_fingertips = None
    
    pipeline, profile = dh.start_d405(exposure='auto')

    all_grip_widths = []
    gripper_state_history = []
    key_states = {}

    start_frames_to_ignore = 15

    gripper_fully_opened = False
    gripper_fingers_touched = False
    
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        if first_frame:
            depth_scale = dh.get_depth_scale(profile)
            print('depth_scale = ', depth_scale)
            print()

            depth_camera_info = dh.get_camera_info(depth_frame)
            color_camera_info = dh.get_camera_info(color_frame)
            print_camera_info = True
            if print_camera_info: 
                for camera_info, name in [(depth_camera_info, 'depth'), (color_camera_info, 'color')]:
                    print(name + ' camera_info:')
                    print(camera_info)
                    print()
            first_frame = False

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        aruco_detection_image = np.copy(color_image)
        #aruco_detector.update(aruco_detection_image, color_camera_info, depth_image, depth_camera_info, depth_scale)
        aruco_detector.update(aruco_detection_image, color_camera_info)
        markers = aruco_detector.get_detected_marker_dict()

        special_origins_3d = []
        fingertips = aruco_to_fingertips.get_fingertips(markers)
        fingers_detected = fingertips.keys()
        both_fingers_detected = ('right' in fingers_detected) and ('left' in fingers_detected)

        distance_between_fingertips = None
        if both_fingers_detected:
            f = fingertips['left']
            fingertip_left_pos = f['pos']
            f = fingertips['right']
            fingertip_right_pos = f['pos']
            grip_width = np.linalg.norm(fingertip_right_pos - fingertip_left_pos)
            all_grip_widths.append(grip_width)
            #print('grip_width =', grip_width)

            if not gripper_fully_opened:
                min_len = 5
                if len(all_grip_widths) > min_len:
                    last_grip_width = all_grip_widths[-1]
                    prev_grip_width = all_grip_widths[-min_len]
                    grip_width_change = last_grip_width - prev_grip_width
                    #print('grip_width_change =', grip_width_change)

                    gripper_state = get_gripper_state(robot)
                    gripper_state['grip_width'] = float(grip_width)
                    gripper_state['command'] = prev_gripper_command
                    gripper_state['fingertips'] = yaml_fingertips(fingertips)

                    gripper_state_history.append(gripper_state)
                    
                    if (grip_width_change < fully_open_fingertip_distance_change) and (len(all_grip_widths) > start_frames_to_ignore):
                        print()
                        print('-----------------------------')
                        print('FINGERS FULLY OPEN')
                        print('stretch_gripper command = ', prev_gripper_command)
                        print('gripper_state =')
                        pp.pprint(gripper_state)
                        print('-----------------------------')
                        key_states['fully_open'] = gripper_state
                        gripper_fully_opened = True
                        all_grip_widths = []
            elif not gripper_fingers_touched:
                min_len = 5
                if len(all_grip_widths) > min_len:
                    gripper_state = get_gripper_state(robot)
                    gripper_state['grip_width'] = float(grip_width)
                    gripper_state['command'] = prev_gripper_command
                    gripper_state['fingertips'] = yaml_fingertips(fingertips)

                    gripper_state_history.append(gripper_state)

                    if (grip_width < just_touching_fingertip_distance) or (prev_gripper_command < -100.0): 
                        print()
                        print('-----------------------------')
                        print('FINGERS JUST TOUCHING')
                        print('stretch_gripper command = ', prev_gripper_command)
                        print('gripper_state =')
                        pp.pprint(gripper_state)
                        print('-----------------------------')
                        key_states['just_touching'] = gripper_state
                        gripper_fingers_touched = True
            else: 
                min_len = 5
                if len(all_grip_widths) > min_len:
                    gripper_state = get_gripper_state(robot)
                    gripper_state['grip_width'] = float(grip_width)
                    gripper_state['command'] = prev_gripper_command
                    gripper_state['fingertips'] = yaml_fingertips(fingertips)

                    gripper_state_history.append(gripper_state)

                    if (prev_gripper_command <= -100.0): 
                        print()
                        print('-----------------------------')
                        print('FINGERS FULLY CLOSED')
                        print('stretch_gripper command = ', prev_gripper_command)
                        print('gripper_state =')
                        pp.pprint(gripper_state)
                        print('-----------------------------')
                        key_states['fully_closed'] = gripper_state

                        results_file_time = time.strftime("%Y%m%d%H%M%S")
                        key_states_filename = './gripper_characterization_key_states_' + results_file_time + '.yaml'
                        state_history_filename = './gripper_characterization_state_history_' + results_file_time + '.yaml'
                        with open(key_states_filename, 'w') as file:
                            yaml.dump(key_states, file, sort_keys=True)
                        print('saved gripper characterization key states to', key_states_filename)
                        with open(state_history_filename, 'w') as file:
                            yaml.dump(gripper_state_history, file, sort_keys=True)
                        print('saved gripper characterization state history to', state_history_filename)

                        exit()
                
            
        fingertip_frames_image = np.copy(color_image)
        aruco_to_fingertips.draw_fingertip_frames(fingertips, fingertip_frames_image, color_camera_info)
        cv2.imshow('Fingertip Frames', fingertip_frames_image)

        if not gripper_fully_opened: 
            new_gripper_command = prev_gripper_command + gripper_change_per_timestep
        else:
            new_gripper_command = prev_gripper_command - gripper_change_per_timestep
            
        if both_fingers_detected and (abs(new_gripper_command - gripper_start) < max_gripper_change): 
            starting_configuration = {
                'stretch_gripper' : new_gripper_command
            }
        
            robot_move.to_configuration(starting_configuration, speed='fast')
            robot.push_command()
            #print('new_gripper_command =', new_gripper_command)
            prev_gripper_command = new_gripper_command

        cv2.waitKey(1)

finally:

    pipeline.stop()
    robot.stop()
