import zmq
import argparse
import forcesight_networking as fn
import argparse
    
def main(use_remote_computer):

    #action_primitives = ['approach', 'grasp', 'lift']
    action_primitives = ['approach', 'grasp']
    
    try: 
        prompt_context = zmq.Context()
        prompt_socket = prompt_context.socket(zmq.PUB)
        prompt_socket.setsockopt(zmq.SNDHWM, 1)
        prompt_socket.setsockopt(zmq.RCVHWM, 1)
        if use_remote_computer:
            address = 'tcp://*:' + str(fn.prompt_port)
        else:
            address = 'tcp://' + '127.0.0.1' + ':' + str(fn.prompt_port)
        prompt_socket.bind(address)
        
        action_status_context = zmq.Context()
        action_status_socket = action_status_context.socket(zmq.SUB)
        action_status_socket.setsockopt(zmq.SUBSCRIBE, b'')
        action_status_socket.setsockopt(zmq.SNDHWM, 1)
        action_status_socket.setsockopt(zmq.RCVHWM, 1)
        action_status_socket.setsockopt(zmq.CONFLATE, 1)    
        action_status_address = 'tcp://' + fn.robot_ip + ':' + str(fn.action_status_port)
        action_status_socket.connect(action_status_address)

        while True:
            use_primitives = input('Use approach and grasp primitives automatically? (Y/N)')
            if ('y' in use_primitives) or ('Y' in use_primitives): 
                prompt = input('Enter a task prompt for ForceSight. Approach and grasp primitives will be appended automatically.:\n')
                primitive_index = 0
                successful = True
                while primitive_index < len(action_primitives):
                    if successful:
                        successful = False
                        current_action = action_primitives[primitive_index]
                        full_prompt = prompt + ', ' + current_action
                        print('Sending new prompt =')
                        print(full_prompt)
                        prompt_socket.send_pyobj(full_prompt)

                    print('waiting to receive from action_status_socket')
                    action_status = action_status_socket.recv_pyobj()
                    print('received action_status =', action_status)
                    current_prompt = action_status['prompt']
                    if current_action in current_prompt:
                        successful = action_status['successful']
                        if successful: 
                            print('Action primitive successful!')
                            primitive_index = primitive_index + 1
                    else:
                        # new action hasn't started
                        successful = False
            else:
                prompt = input('Enter a full prompt for ForceSight. No primitives will be appended.:\n')
                print('Sending new prompt =')
                print(prompt)
                prompt_socket.send_pyobj(prompt)
            
    finally:
        pipeline.stop()

if __name__ == '__main__':
    use_remote_computer = False
    main(use_remote_computer)
    
