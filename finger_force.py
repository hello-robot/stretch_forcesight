import time
import yaml
from yaml.loader import SafeLoader
import numpy as np
import argparse
import glob

lut_filename_prefix = './gripper_position_lookup_table_'
lut_filename_suffix = '.yaml'

class FingerForce:
    def __init__(self):
        self.pos_array = None
    
    
    def load_lookup_table(self):
        lut_filename_pattern = lut_filename_prefix + '*' + lut_filename_suffix
        filenames = glob.glob(lut_filename_pattern)
        if len(filenames)>0:
            filename = filenames[-1]
        else:
            print(f'Exiting now, since no file was found file matching the following pattern: {lut_filename_pattern}. Have you run the gripper characterization code and then the lookup table code to generate this type of file?')
            exit()
        print('file to load =', filename)

        with open(filename) as f:
            lookup_table = yaml.load(f, Loader=SafeLoader)

        self.pos_array = np.array(lookup_table)
            

    def prepare_lookup_table(self):
    
        np.set_printoptions(precision=3, linewidth=100, suppress=True)

        state_history_filename_pattern = './gripper_characterization_state_history_*.yaml'
        filenames = glob.glob(state_history_filename_pattern)
        if len(filenames)>0:
            filename = filenames[-1]
        else:
            print(f'Exiting now, since no file was found file matching the following pattern: {state_history_filename_pattern}. Have you run the gripper characterization code to generate this type of file?')
            exit()
        print('file to load =', filename)


        with open(filename) as f:
            states = yaml.load(f, Loader=SafeLoader)

        #print('loaded gripper state history')
        #print(f'len(states) = {len(states)}')

        # Extract the gripper positions

        grip_pos = []
        fingertip_pos = []
        for s in states:
            grip_pos.append(s['pos'])
            left_pos = s['fingertips']['left']['pos']
            right_pos = s['fingertips']['right']['pos']
            fingertip_pos.append([left_pos, right_pos])

        positions = [[g] + f[0] + f[1] for g,f in sorted(zip(grip_pos, fingertip_pos))]
        #print(f'positions = {positions}')
        self.pos_array = np.array(positions)
        #print(f'self.pos_array.shape = {self.pos_array.shape}')
        #print('self.pos_array =')
        #print(self.pos_array)

        
    def lookup_fingertip_positions(self, grip_pos):
        assert self.pos_array is not None, 'ERROR: FingerForce.lookup_fingertip_positions called before preparing or loading a lookup table.'

        positions = self.pos_array[:,0]
        #print('positions =')
        #print(positions)
        index = np.searchsorted(positions, grip_pos)
        #print(f'index before bounds checking = {index}')
        #print(f'self.pos_array.shape = {self.pos_array.shape}')
        if index < 0:
            index = 0
            print('WARNING: index too small and out of bounds, so clipped')
        lut_len = self.pos_array.shape[0]
        if index >= lut_len:
            print(f'WARNING: index = index is too big and out of bounds with lut_len = {lut_len}, so clipped')
            index = -1
        #print(f'search: pos = {grip_pos}, index = {index}')
        #print(f'array row = {self.pos_array[index,:]}')
        row = self.pos_array[index,:]
        return row


    def finger_position_difference(self, grip_pos, fingertips):
        row = self.lookup_fingertip_positions(grip_pos)
        def get_diff(side): 
            f = fingertips.get(side, None)
            if f is not None:
                if side == 'left': 
                    diff = row[1:4] - f['pos']
                elif side == 'right':
                    diff = row[4:] - f['pos']
                return diff
            else:
                return None
        side = 'left'
        left_diff = get_diff('left')
        right_diff = get_diff('right')
        return {'left_diff': left_diff, 'right_diff': right_diff}


    def gripper_forces(self, grip_pos, fingertips):
        k_down = 10.0/-0.03 # N/m
        k_out = -10.0/0.017 # N/m
        k_back = 20.0/0.02 # N/m
        K = np.array([k_out, k_down, k_back])

        left_force = None
        right_force = None
        diff = self.finger_position_difference(grip_pos, fingertips)
        left_diff = diff['left_diff']
        if left_diff is not None:
            left_force = K * left_diff
        right_diff = diff['right_diff']
        if right_diff is not None:
            right_force = K * right_diff
        return {'left_force': left_force, 'right_force': right_force}


    def forcesight_forces(self, grip_pos, fingertips):
        gripper_forces = self.gripper_forces(grip_pos, fingertips)
        applied_force = None
        grip_force = None
        left_force = gripper_forces['left_force']
        right_force = gripper_forces['right_force']
        if (left_force is not None) and (right_force is not None):
            # grip with high compressive force  => positive grip force
            # grip in tension due to adhesion => negative grip force
            # push up => negative y
            # push to robot's right => positive x
            # push in => positive z (insensitive with high error
            applied_force = left_force + right_force
            grip_force = applied_force[0] - left_force[0]
            # zero out the applied force before starting or maybe just
            # the z axis applied force?
        return {'grip_force': grip_force, 'applied_force': applied_force}

    
    def forcesight_forces_using_efforts(self, grip_pos, fingertips, pitch_eff, yaw_eff, arm_eff, non_contact_pitch_eff):
        gripper_forces = self.gripper_forces(grip_pos, fingertips)
        applied_force = None
        grip_force = None
        left_force = gripper_forces['left_force']
        right_force = gripper_forces['right_force']
        if (left_force is not None) and (right_force is not None):
            # grip with high compressive force  => positive grip force
            # grip in tension due to adhesion => negative grip force
            # push up => negative y
            # push to robot's right => positive x
            # push in => positive z (insensitive with high error
            applied_force = left_force + right_force
            grip_force = applied_force[0] - left_force[0]
            print('pitch_eff =', pitch_eff)
            applied_force[2] = -arm_eff
            applied_force[1] = -(4.0/5.0) * (pitch_eff - non_contact_pitch_eff)
            applied_force[0] = -yaw_eff
            # zero out the applied force before starting or maybe just
            # the z axis applied force?
        return {'grip_force': grip_force, 'applied_force': applied_force}
    
    def save_lookup_table(self):
        results_file_time = time.strftime("%Y%m%d%H%M%S")
        lut_filename = lut_filename_prefix + results_file_time + lut_filename_suffix
        with open(lut_filename, 'w') as file:
            yaml.dump(self.pos_array.tolist(), file)
            print('saved gripper position lookup table to', lut_filename)

        
if __name__ == '__main__':
    ff = FingerForce()
    ff.prepare_lookup_table()
    ff.save_lookup_table()
    
