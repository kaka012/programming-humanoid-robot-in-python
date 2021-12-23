'''In this file you need to implement remote procedure call (RPC) server

* There are different RPC libraries for python, such as xmlrpclib, json-rpc. You are free to choose.
* The following functions have to be implemented and exported:
 * get_angle
 * set_angle
 * get_posture
 * execute_keyframes
 * get_transform
 * set_transform
* You can test RPC server with ipython before implementing agent_client.py
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'kinematics'))

from inverse_kinematics import InverseKinematicsAgent
import time
import xmlrpc.server
import threading
import numpy as np

class ServerAgent(InverseKinematicsAgent):
    '''ServerAgent provides RPC service
    '''
    def __init__(self, *args, **kwargs):
        super(ServerAgent, self).__init__(*args, **kwargs)

        t = threading.Thread(target=self.handle_rpcs)
        t.daemon = True
        t.start()

    def handle_rpcs(self):
        with xmlrpc.server.SimpleXMLRPCServer(("localhost", 8000), allow_none=True) as server:
            server.register_function(self.get_angle)
            server.register_function(self.set_angle)
            server.register_function(self.get_posture)
            server.register_function(self.execute_keyframes)
            server.register_function(self.get_transform)
            server.register_function(self.set_transform)
            server.serve_forever()

    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        return self.sensor_joints[joint_name]

    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        self.target_joints[joint_name] = angle

    def get_posture(self):
        '''return current posture of robot'''
        return self.posture

    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        while len(self.keyframes[0]) > 0:
            time.sleep(1/100)
        self.keyframes = keyframes
        while len(self.keyframes[0]) > 0:
            time.sleep(1/100)

    def get_transform(self, name):
        '''get transform with given name
        '''
        return self.transforms[name].tolist()

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        self.set_transforms(effector_name, np.array(transform, np.float32))

if __name__ == '__main__':
    agent = ServerAgent()
    agent.run()

