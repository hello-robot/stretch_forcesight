import zmq
import numpy as np
import cv2
import aruco_detector as ad
import aruco_to_fingertips as af
import yaml
from yaml.loader import SafeLoader
import d405_helpers_without_pyrealsense as dh
import loop_timer as lt
import forcesight_networking as fn
import argparse
from pprint import pprint

import forcesight_min as fs


def draw_text(image, origin, text_lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    location = origin + np.array([0, -55])
    location = location.astype(np.int32)
        
    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, font, font_size, 4)
        (text_width, text_height), text_baseline = text_size
        center = int(text_width / 2)
        offset = np.array([-center, i * (1.7*text_height)]).astype(np.int32)
        cv2.putText(image, line, location + offset, font, font_size, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, location + offset, font, font_size, (255, 255, 255), 1, cv2.LINE_AA)


def main(use_remote_computer):

    #confidence_threshold = 0.005
    
    forcesight = fs.ForceSightMin(fn.forcesight_model)
    
    marker_info = {}
    with open('aruco_marker_info.yaml') as f:
        marker_info = yaml.load(f, Loader=SafeLoader)
    aruco_detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=False, use_apriltag_refinement=False, brighten_images=False)
    fingertip_part = 'cup_top' #'cup_bottom'
    aruco_to_fingertips = af.ArucoToFingertips(default_height_above_mounting_surface=af.suctioncup_height[fingertip_part])
    
    forcesight_context = zmq.Context()
    forcesight_socket = forcesight_context.socket(zmq.PUB)
    if use_remote_computer:
        forcesight_address = 'tcp://*:' + str(fn.forcesight_port)
    else:
        forcesight_address = 'tcp://' + '127.0.0.1' + ':' + str(fn.forcesight_port)
    forcesight_socket.setsockopt(zmq.SNDHWM, 1)
    forcesight_socket.setsockopt(zmq.RCVHWM, 1)
    forcesight_socket.bind(forcesight_address)
    
    d405_context = zmq.Context()
    d405_socket = d405_context.socket(zmq.SUB)
    d405_socket.setsockopt(zmq.SUBSCRIBE, b'')
    d405_socket.setsockopt(zmq.SNDHWM, 1)
    d405_socket.setsockopt(zmq.RCVHWM, 1)
    d405_socket.setsockopt(zmq.CONFLATE, 1)
    if use_remote_computer:
        d405_address = 'tcp://' + fn.robot_ip + ':' + str(fn.d405_port)
    else:
        d405_address = 'tcp://' + '127.0.0.1' + ':' + str(fn.d405_port)
    d405_socket.connect(d405_address)

    prompt_context = zmq.Context()
    prompt_socket = prompt_context.socket(zmq.SUB)
    prompt_socket.setsockopt(zmq.SUBSCRIBE, b'')
    prompt_socket.setsockopt(zmq.SNDHWM, 1)
    prompt_socket.setsockopt(zmq.RCVHWM, 1)
    prompt_socket.setsockopt(zmq.CONFLATE, 1)
    # For now, run the prompt server on the same machine.
    prompt_address = 'tcp://' + '127.0.0.1' + ':' + str(fn.prompt_port)
    prompt_socket.connect(prompt_address)

    prompt = "Do Nothing"
    
    poller = zmq.Poller()
    poller.register(d405_socket, zmq.POLLIN)
    poller.register(prompt_socket, zmq.POLLIN)
    
    loop_timer = lt.LoopTimer()
    
    try:

        first_frame = True
        
        while True:

            loop_timer.start_of_iteration()
            
            sockets = dict(poller.poll())

            new_images = False
            if d405_socket in sockets and sockets[d405_socket] == zmq.POLLIN:
                d405_output = d405_socket.recv_pyobj()
                color_image = d405_output['color_image']
                depth_image = d405_output['depth_image']
                depth_camera_info = d405_output['depth_camera_info']
                color_camera_info = d405_output['color_camera_info']
                depth_scale = d405_output['depth_scale']
                camera_info = depth_camera_info
                forcesight.set_camera_info(camera_info)
                new_images = True
                

            if prompt_socket in sockets and sockets[prompt_socket] == zmq.POLLIN:
                prompt = prompt_socket.recv_pyobj()

            forcesight_output = None
            if new_images:
                aruco_detector.update(color_image, camera_info)
                markers = aruco_detector.get_detected_marker_dict()
                fingertips = aruco_to_fingertips.get_fingertips(markers)

                prediction = forcesight.apply(color_image,
                                              depth_image,
                                              prompt,
                                              camera_info,
                                              depth_scale)

                if True: 
                    #################################################
                    # This attempts to correct for differences between the
                    # robot used for training ForceSight and a Stretch 3
                    # robot.
                    grip_center_correction = np.array([0.0, 0.0, -0.04])
                    #grip_center_correction = np.array([0.0, 0.0, -0.035])
                    #grip_center_correction = np.array([0.0, 0.0, -0.03])
                    grip_center = prediction['grip_center']['xyz_m']
                    if grip_center is not None:
                        prediction['grip_center']['xyz_m'] = grip_center + grip_center_correction
                    left_fingertip = prediction['fingertips']['left']['xyz_m']
                    if left_fingertip is not None:
                        prediction['fingertips']['left']['xyz_m'] = left_fingertip + grip_center_correction
                    right_fingertip = prediction['fingertips']['right']['xyz_m']
                    if right_fingertip is not None:
                        prediction['fingertips']['right']['xyz_m'] = right_fingertip + grip_center_correction
                    #################################################
                    
                #confidence = prediction['confidence']
                #if (confidence is not None) and (confidence > confidence_threshold):
                forcesight_output = forcesight.prediction_without_images(prediction)

                grip_force = forcesight_output['grip_force_n']
                applied_force = forcesight_output['applied_xyz_force_n']
                applied_force_camera = forcesight_output['applied_force_camera_n']

                print('grip_force =', grip_force)
                print('applied_force =', applied_force)
                print('applied_force_camera =', applied_force_camera)
                
                
                send_dict = {
                    'fingertips': fingertips,
                    'forcesight': forcesight_output
                } 

                forcesight_socket.send_pyobj(send_dict)

                output_image = np.copy(color_image)
                aruco_to_fingertips.draw_fingertip_frames(fingertips,
                                                          output_image,
                                                          camera_info,
                                                          axis_length_in_m=0.02,
                                                          draw_origins=True,
                                                          write_coordinates=True)

                #forcesight.draw_prediction(output_image, prediction, confidence_threshold, show_depth=False)
                forcesight.draw_prediction(output_image, prediction, show_depth=False)

                display_grasp_center_and_distance_images = False
                if display_grasp_center_and_distance_images:
                    forcesight.show_prediction_images(prediction)

                display_depth_image = False
                if display_depth_image:
                    colorized_depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    cv2.imshow('Depth Image', colorized_depth_image)


                grip_center = prediction['grip_center']['xyz_m']
                grip_width = prediction['grip_width_m']
                if (grip_center is not None) and (grip_width is not None): 

                    grasp_center_xy = dh.pixel_from_3d(grip_center, camera_info)
                    grasp_point = grasp_center_xy.astype(np.int32)
                    radius = 6
                    cv2.circle(output_image, grasp_point, radius, (255, 0, 0), -1, lineType=cv2.LINE_AA)

                    x,y,z = grip_center * 100.0
                    text_lines = [
                        "{:.1f} cm wide".format(grip_width * 100.0),
                        "{:.1f}, {:.1f}, {:.1f} cm".format(x,y,z)
                        ]
                    draw_text(output_image, grasp_point, text_lines)

                cv2.imshow('ForceSight Visualization', output_image)
                
                
            cv2.waitKey(1)

            loop_timer.end_of_iteration()
            loop_timer.pretty_print()
    
    finally:
        pass


if __name__ == '__main__':

    main(use_remote_computer=True)
