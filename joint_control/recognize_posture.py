'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''


from angle_interpolation import AngleInterpolationAgent
from keyframes import leftBellyToStand, leftBackToStand
from collections import deque
import pickle
import os


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.posture = 'unknown'
        self.posture_confidence = 0.0
        self.last_posture_ests = deque([''], maxlen=5)
        self.posture_classifier = pickle.load(open('joint_control/robot_pose.pkl', 'rb'))

    def think(self, perception):
        posture_est = self.recognize_posture(perception)
        self.last_posture_ests.append(posture_est)
        # The estimator isn't perfect, so wait until we have a consistent
        # estimate before declaring it as correct. This adds some delay, but
        # also some reliability.
        self.posture_confidence = sum([
                1 if p == posture_est else 0
                for p in self.last_posture_ests
        ]) / len(self.last_posture_ests)
        if self.posture_confidence >= 1.0:
            self.posture = posture_est
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        postures = ['Back', 'Belly', 'Crouch', 'Frog', 'HeadBack', 'Knee',
                    'Left', 'Right', 'Sit', 'Stand', 'StandInit']
        features = ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch',
                    'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch',
                    'AngleX', 'AngleY']
        imu_features = ['AngleX', 'AngleY']

        input_data = []
        for feature in features:
            if feature in imu_features:
                input_data.append(perception.imu[imu_features.index(feature)])
            else:
                input_data.append(perception.joint[feature])

        return postures[self.posture_classifier.predict([input_data])[0]]


if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = leftBackToStand()
    agent.run()
