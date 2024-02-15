import time
import yaml
from yaml.loader import SafeLoader
import numpy as np
import argparse
import glob

def main():

    lookup_table_size = 100

    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    
    state_history_filename = './gripper_characterization_state_history_*.yaml'
    filenames = glob.glob(state_history_filename)
    if len(filenames)>0:
        filename = filenames[-1]
    else:
        print(f'Exiting now, since no file was found file matching the following pattern: {state_history_filename}. Have you run the gripper characterization code to generate this type of file?')
        exit()
    print('file to load =', filename)

    
    with open(filename) as f:
        states = yaml.load(f, Loader=SafeLoader)

    print('loaded gripper state history')
    print(f'len(states) = {len(states)}')

    # Extract the gripper positions

    grip_pos = []
    fingertip_pos = []
    for s in states:
        grip_pos.append(s['pos'])
        left_pos = s['fingertips']['left']['pos']
        right_pos = s['fingertips']['right']['pos']
        fingertip_pos.append([left_pos, right_pos])

    positions = [[g] + f[0] + f[1] for g,f in sorted(zip(grip_pos, fingertip_pos))]
    print(f'positions = {positions}')
    pos_array = np.array(positions)
    print(f'pos_array.shape = {pos_array.shape}')
    print('pos_array =')
    print(pos_array)

    pos = 9.1
    index = np.searchsorted(pos_array[:,0], pos)
    print(f'search: pos = {pos}, index = {index}')
    print(f'array row = {pos_array[index,:]}')

    results_file_time = time.strftime("%Y%m%d%H%M%S")
    lut_filename = './gripper_position_lookup_table_' + results_file_time + '.yaml'
    with open(lut_filename, 'w') as file:
        yaml.dump(pos_array.tolist(), file)
        print('saved gripper position lookup table to', lut_filename)

if __name__ == '__main__':
    main()
    
