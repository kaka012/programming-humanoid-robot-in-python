'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

import numpy as np
import sympy

from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: np.identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll'],
                       'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll'],
                       'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                       'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'],
                       }
        # http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
        # Defined in mm
        links = {
            'HeadYaw':        (  0.00,  0.00,  126.50),
            'HeadPitch':      (  0.00,  0.00,    0.00),
            'LShoulderPitch': (  0.00, 98.00,  100.00),
            'LShoulderRoll':  (  0.00,  0.00,    0.00),
            'LElbowYaw':      (105.00, 15.00,    0.00),
            'LElbowRoll':     (  0.00,  0.00,    0.00),
            'LWristYaw':      ( 55.95,  0.00,    0.00),
            'LHipYawPitch':   (  0.00, 50.00,  -85.00),
            'LHipRoll':       (  0.00,  0.00,    0.00),
            'LHipPitch':      (  0.00,  0.00,    0.00),
            'LKneePitch':     (  0.00,  0.00, -100.00),
            'LAnklePitch':    (  0.00,  0.00, -102.90),
            'LAnkleRoll':     (  0.00,  0.00,    0.00),
        }
        # mm to m
        self.links = {k: tuple([i / 1000 for i in v]) for k, v in links.items()}
        # Mirror the links.
        for k, v in list(self.links.items()):
            if k[0] == 'L':
                self.links[f"R{k[1:]}"] = (v[0], -v[1], v[2])
        # Rotation axes
        joint_axes = {
            'HeadYaw':        (0, 0, 1),
            'HeadPitch':      (0, 1, 0),
            'LShoulderPitch': (0, 1, 0),
            'LShoulderRoll':  (0, 0, 1),
            'LElbowYaw':      (1, 0, 0),
            'LElbowRoll':     (0, 0, 1),
            'LWristYaw':      (1, 0, 0),
            'LHipYawPitch':   (0, 1,-1),
            'RHipYawPitch':   (0, 1, 1),
            'LHipRoll':       (1, 0, 0),
            'LHipPitch':      (0, 1, 0),
            'LKneePitch':     (0, 1, 0),
            'LAnklePitch':    (0, 1, 0),
            'LAnkleRoll':     (1, 0, 0),
        }
        # Mirror the rotation axes, but only for "Roll" joints.
        for k, v in list(joint_axes.items()):
            if k[0] == 'L' and not 'HipYawPitch' in k:
                x, y, z = v
                if 'Roll' in k:
                    y = -y
                joint_axes[f"R{k[1:]}"] = (x, y, z)
        self.joint_axes = joint_axes

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    @staticmethod
    def normalize_axis_vector(axis, dtype=None):
        v = np.array([
            [axis[0]],
            [axis[1]],
            [axis[2]],
            [0]
        ], dtype)
        return v / np.linalg.norm(v)

    def local_trans(self, joint_name, joint_angle, symbolic=False):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        if symbolic:
            dtype = None
            sin, cos = sympy.sin, sympy.cos
        else:
            dtype = np.float32
            sin, cos = np.sin, np.cos

        r = ForwardKinematicsAgent.normalize_axis_vector(self.joint_axes[joint_name], dtype)

        # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        phi = joint_angle
        m = r * r.T * (1 - cos(phi))
        m += np.array([
            [           cos(phi), -r[2, 0] * sin(phi),  r[1, 0] * sin(phi), 0],
            [ r[2, 0] * sin(phi),            cos(phi), -r[0, 0] * sin(phi), 0],
            [-r[1, 0] * sin(phi),  r[0, 0] * sin(phi),            cos(phi), 0],
            [                  0,                   0,                   0, 1],
        ], dtype)

        m += np.array([
            [0, 0, 0, self.links[joint_name][0]],
            [0, 0, 0, self.links[joint_name][1]],
            [0, 0, 0, self.links[joint_name][2]],
            [0, 0, 0,                         0],
        ], dtype)

        return m

    def forward_kinematics(self, joints, symbolic=False):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        transforms = {n: np.identity(4) for n in self.joint_names}
        for chain_joints in self.chains.values():
            T = np.identity(4, np.float32)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle, symbolic)
                T = T @ Tl

                transforms[joint] = T
        return transforms

# Visualization to verify each joints rotation direction is correct
# Shown are:
# - All chains in random (but deterministic) colors
# - All local transformation directions (x red, y green, z blue) of each joint
# - All local rotation axes in black of each joint
def draw(i, ax, agent, transforms={}, anim_angles=False):
    global anim_data

    import itertools

    if i == 0:
        ax.clear()
        ax.set(xlim=(-0.225, 0.225), ylim=(-0.225, 0.225), zlim=(-0.30, 0.15))
        anim_data = {}

    if anim_angles:
        # Choose which joints to animate and how fast/far.
        d = {
            'HeadYaw':        0.0,
            'HeadPitch':      0.0,
            'LShoulderPitch': 0.0,
            'LShoulderRoll':  0.0,
            'LElbowYaw':      0.0,
            'LElbowRoll':     0.0,
            'LWristYaw':      0.0,
            'LHipYawPitch':   0.5,
            'LHipRoll':       0.0,
            'LHipPitch':      0.0,
            'LKneePitch':     0.0,
            'LAnklePitch':    0.0,
            'LAnkleRoll':     0.0,
        }
        for k in list(d.keys()):
            if k[0] == 'L':
                d[f"R{k[1:]}"] = d[k]
        # Linearly go from -pi/2 (-90°) to pi/2 (90°) in 32 steps.
        for k in d.keys():
            d[k] *= -np.pi / 2 + (i % 33) * np.pi / 32

        transforms = agent.forward_kinematics(d)
    if len(transforms) == 0:
        transforms = agent.transforms

    p_data = {}
    t_ax_len = 0.05
    q_ax_len = 0.07
    j_line_width = 5
    t_line_width = 2
    q_line_width = 1
    for chain_joints in agent.chains.values():
        old_P = np.array([0, 0, 0], np.float32)
        for joint in chain_joints:
            P = transforms[joint] @ np.array([[0], [0], [0], [1]], np.float32)
            Px = t_ax_len * (transforms[joint] @ np.array([[1], [0], [0], [0]], np.float32))
            Py = t_ax_len * (transforms[joint] @ np.array([[0], [1], [0], [0]], np.float32))
            Pz = t_ax_len * (transforms[joint] @ np.array([[0], [0], [1], [0]], np.float32))
            r = ForwardKinematicsAgent.normalize_axis_vector(agent.joint_axes[joint], np.float32)
            Pq = q_ax_len * (transforms[joint] @ r)

            if i == 0:
                p_data[joint] = [zip(old_P, P[:3, 0])] + \
                    [zip(P[:3, 0], (P + p)[:3, 0]) for p in [Pq, Px, Py, Pz]]
            else:
                anim_data[joint][0].set_data_3d(*zip(old_P, P[:3, 0]))
                for i, p in enumerate((Pq, Px, Py, Pz), 1):
                    anim_data[joint][i].set_data_3d(*zip(P[:3, 0], (P + p)[:3, 0]))
            old_P = P[:3, 0]
    if i != 0:
        return
    # Draw all links first
    line_widths = [j_line_width] + [q_line_width] + [t_line_width] * 3
    for joint in itertools.chain.from_iterable(agent.chains.values()):
        anim_data[joint] = [ax.plot(*p_data[joint][0], linewidth=j_line_width)[0]]
    # And only then the transformations, so they are on top.
    for joint in itertools.chain.from_iterable(agent.chains.values()):
        for i, color in enumerate(('k-', 'r-', 'g-', 'b-'), 1):
            anim_data[joint] += [ax.plot(*p_data[joint][i], color, linewidth=line_widths[i])[0]]

def play_animation():
    global ax, agent

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ani = FuncAnimation(fig, draw, frames=128, interval=50, repeat=True,
                        fargs=(ax, agent, {}, True))
    plt.show(block=True)

def print_checks():
    global ax, agent
    
    # Load data
    import json
    with open('kinematics/fk_samples.json') as f:
        d = json.load(f)
    for pose in ['init', 'moveInit', 'rest']:
        joint_angles = {
            joint_name: d[pose]['angles'][i]
            for i, joint_name in enumerate(d['joint_names'])
        }
        # Evaluate data
        agent.transforms = agent.forward_kinematics(joint_angles)

        # Plot data
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        draw(0, ax, agent)
        for p in d[pose]['positions']:
            ax.plot(*p[:3], 'cx')
        plt.show(block=True)

        # Check data
        import transforms3d as tf
        for k, v in joint_angles.items():
            print(f"{k}: {v}")
        for joint_name in agent.transforms:
            print(joint_name)
            print(agent.transforms[joint_name])
            print(agent.transforms[joint_name][:3,3].flatten())
            T, R, Z, S = tf.affines.decompose44(agent.transforms[joint_name])
            R = tf.euler.mat2euler(R)
            print(f"T: {T}\nR: {R}\nZ: {Z}\nS: {S}")
            if joint_name in d['names']:
                TR_ref = np.array(d[pose]['positions'][d['names'].index(joint_name)])
                TR = np.array(list(T) + list(R))
                print(f">>> {TR_ref}")
                dt = np.linalg.norm(TR[:3] - TR_ref[:3])
                dr = sum(abs(TR[:3] - TR_ref[:3]))
                weighted_delta = dt + dr / 10

                if weighted_delta > 10 ** -3:
                    print(f">>> Mismatch (delta {weighted_delta:.9f})!")
                else:
                    print(f">>> Match (delta {weighted_delta:.9f})!")

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    print_checks()
    play_animation()
