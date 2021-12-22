'''In this exercise you need to put all code together to make the robot be able to stand up by its own.

* Task:
    complete the `StandingUpAgent.standing_up` function, e.g. call keyframe motion corresponds to current posture

'''


from recognize_posture import PostureRecognitionAgent
from keyframes import leftBackToStand, leftBellyToStand, rightBackToStand, rightBellyToStand
import random

class StandingUpAgent(PostureRecognitionAgent):
    def think(self, perception):
        self.standing_up(perception)
        return super(StandingUpAgent, self).think(perception)

    def standing_up(self, perception):
        posture = self.posture
        if self.animation_running:
            return
        if perception.time - self.stiffness_on_off_time < self.stiffness_off_cycle \
            or perception.time - self.stiffness_on_off_time - self.stiffness_on_cycle + self.stiffness_off_cycle > 0.5:
            self.joint_controller.set_enabled(False)
            return
        self.joint_controller.set_enabled(True)
        if posture == 'Back':
            keyframe_fun = random.choice([leftBackToStand, rightBackToStand])
        elif posture == 'Belly':
            keyframe_fun = random.choice([leftBellyToStand, rightBellyToStand])
        else:
            return
        print(f"Preparing animation '{keyframe_fun.__name__}'")
        names, times, keys = keyframe_fun()
        _, ctimes, ckeys = self.joints_as_keyframe(perception.joint, 0.5, names)
        if posture == 'Belly':
            keys[names.index('LShoulderPitch')][0] = ckeys[names.index('LShoulderPitch')][0]
            keys[names.index('RShoulderPitch')][0] = ckeys[names.index('RShoulderPitch')][0]
        keyframes = (names, times, keys)
        keyframe = (names, ctimes, ckeys)
        keyframe = self.animation_transform(keyframe, 0)
        keyframes = self.animation_transform(keyframes, 1)
        self.keyframes = self.animations_concat(keyframe, keyframes)

class TestStandingUpAgent(StandingUpAgent):
    '''this agent turns off all motor to falls down in fixed cycles
    '''
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(TestStandingUpAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.stiffness_on_off_time = 0
        self.stiffness_on_cycle = 10  # in seconds
        self.stiffness_off_cycle = 3  # in seconds

    def think(self, perception):
        action = super(TestStandingUpAgent, self).think(perception)
        time_now = perception.time
        if time_now - self.stiffness_on_off_time < self.stiffness_off_cycle:
            action.stiffness = {j: 0 for j in self.joint_names}  # turn off joints
        else:
            action.stiffness = {j: 1 for j in self.joint_names}  # turn on joints
        if time_now - self.stiffness_on_off_time > self.stiffness_on_cycle + self.stiffness_off_cycle:
            self.stiffness_on_off_time = time_now

        return action


if __name__ == '__main__':
    agent = TestStandingUpAgent()
    agent.run()
