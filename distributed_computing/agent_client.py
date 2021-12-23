'''In this file you need to implement remote procedure call (RPC) client

* The agent_server.py has to be implemented first (at least one function is implemented and exported)
* Please implement functions in ClientAgent first, which should request remote call directly
* The PostHandler can be implement in the last step, it provides non-blocking functions, e.g. agent.post.execute_keyframes
 * Hints: [threading](https://docs.python.org/2/library/threading.html) may be needed for monitoring if the task is done
'''

import weakref
import xmlrpc.client
import threading
import numpy as np

class PostHandler(object):
    '''the post handler wraps functions to be executed in parallel
    '''
    def __init__(self, obj):
        self.proxy = weakref.proxy(obj)

    def execute_keyframes(self, keyframes):
        '''non-blocking call of ClientAgent.execute_keyframes'''
        t = threading.Thread(target=self.proxy.execute_keyframes, args=(keyframes,))
        t.start()
        return t

    def set_transform(self, effector_name, transform):
        '''non-blocking call of ClientAgent.set_transform'''
        t = threading.Thread(target=self.proxy.set_transform, args=(effector_name, transform))
        t.start()
        return t


class ClientAgent(object):
    '''ClientAgent request RPC service from remote server
    '''
    def __init__(self):
        self.post = PostHandler(self)
        self.s = xmlrpc.client.ServerProxy("http://localhost:8000/")

    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        return self.s.get_angle(joint_name)

    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        return self.s.set_angle(joint_name, angle)

    def get_posture(self):
        '''return current posture of robot'''
        return self.s.get_posture()

    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        self.s.execute_keyframes(keyframes)

    def get_transform(self, name):
        '''get transform with given name
        '''
        return np.array(self.s.get_transform(name), np.float32)

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        self.s.set_transform(effector_name, transform.tolist())

if __name__ == '__main__':
    import time
    agent = ClientAgent()
    angle = agent.get_angle("HeadYaw")
    print(f"HeadYaw: {angle}")
    if angle > 0:
        agent.set_angle("HeadYaw", -0.785)
    else:
        agent.set_angle("HeadYaw", 0.785)
    print(f"Posture: {agent.get_posture()}")
    agent.execute_keyframes((["HeadPitch"], [[0.0, 1.0, 2.0]], [[[0.0, [3, -0.1, 0.1], [3, -0.1, 0.1]], [-0.5, [3, -0.1, 0.1], [3, -0.1, 0.1]], [0.0, [3, -0.1, 0.1], [3, -0.1, 0.1]]]]))
    print(f"HeadPitch transform:\n{agent.get_transform('HeadPitch')}")

    fps = 30
    for t in np.linspace(0.0, 2.0, 2 * fps + 1):
        scale = 0.1
        z = (1.0 - abs(t - 1.0)) * scale
        T = np.identity(4)
        pos = [0, 0.05, -0.330 + z] # x, y, z
        T[0, -1] = pos[0]
        T[1, -1] = pos[1]
        T[2, -1] = pos[2]
        agent.set_transform("LLeg", T)
        T[1, -1] = -pos[1]
        agent.set_transform("RLeg", T)
