import sys
import os
import numpy as np
from vrepper.vrepper import vrepper
from vrepper.vrepper import vrep
import math
# взаимодействие с v-rep
class Q_learning():
    def __init__(self, headless=True):
        vrep.simxFinish(-1)
        self.venv = vrepper(headless=headless, dir_vrep="C:/Program Files/V-REP3/V-REP_PRO_EDU/")
        self.venv.start()
        self.venv.load_scene(os.path.dirname(os.getcwd()) + '/scenes/diploma.ttt')
        self.sensors = ['sensor_leftleft', 'sensor_left', 'sensor_front', 'sensor_right', 'sensor_rightright']
        self.motors = ['motor_left_1', 'motor_left_2', 'motor_left_3', 'motor_right_1', 'motor_right_2',
                       'motor_right_3']
        self.robot = self.get_object_handles()
        self.position = self.get_position()
        self.set_random_target()
        self.prev_distance = np.sqrt(np.sum(np.power(self.dest - self.position, 2)))
        self.distance = np.copy(self.prev_distance)
        self.prev_angle = self.angle_between(self.position, self.dest, self.get_orientation(axis=2))
        self.angle = np.copy(self.prev_angle)
        self.previous_speed = np.array([0, 0])
        self.venv.start_blocking_simulation()

    def angle_between(self, position, destination, orientation):
        otn_k = destination - position
        otn_angle = np.arctan2(otn_k[1], otn_k[0])
        angle_error = (otn_angle - orientation) / math.pi
        if angle_error < -1:
            angle_error += 2.
        if angle_error > 1:
            angle_error -= 2.
        return angle_error

    def get_object_handles(self):
        robot = {}
        vrep_parts = ['Caterpillar_respondable', 'Proximity_sensor_leftleft', 'Proximity_sensor_left',
                      'Proximity_sensor_front', 'Proximity_sensor_right', 'Proximity_sensor_rightright',
                      'dynamicLeftJoint1', 'dynamicLeftJoint2', 'dynamicLeftJoint3',
                      'dynamicRightJoint1', 'dynamicRightJoint2', 'dynamicRightJoint3']
        for i, part in enumerate(['body'] + self.sensors + self.motors):
            robot[part] = vrep.simxGetObjectHandle(self.venv.cid, vrep_parts[i], vrep.simx_opmode_blocking)[1]
        return robot

    def read_data(self, step):
        data = np.zeros(9)
        readings = np.zeros((len(self.sensors), 3))
        for i, sensor in enumerate(self.sensors):
            readings[i] = np.asarray(vrep.simxReadProximitySensor(self.venv.cid, self.robot[sensor],
                                                                  vrep.simx_opmode_blocking)[2])
        data[:len(self.sensors)] = np.sqrt(np.sum(np.power(readings, 2), axis=1))
        motors = ['motor_left_1', 'motor_right_1']
        data[5:7] = self.previous_speed
        self.position = self.get_position()
        if step:
            self.prev_distance = np.copy(self.distance)
            self.prev_angle = np.copy(self.angle)
            data[-2] = self.distance = np.sqrt(np.sum(np.power(self.dest - self.position, 2)))
            data[-1] = self.angle = self.angle_between(self.position, self.dest, self.get_orientation(axis=2))
        else:
            data[-2] = np.sqrt(np.sum(np.power(self.dest - self.position, 2)))
            data[-1] = self.angle_between(self.position, self.dest, self.get_orientation(axis=2))
        return data

    def send_data(self, speed):
        speed = -10. * speed
        # motors = np.zeros(2)
        # motors[0] = np.clip(speed[0] - speed[1], a_min=-1, a_max=1)
        # motors[1] = np.clip(speed[0] + speed[1], a_min=-1, a_max=1)
        # motors = -3. * motors
        for i, motor in enumerate(self.motors):
            vrep.simxSetJointTargetVelocity(self.venv.cid, self.robot[motor], speed[i // 3], vrep.simx_opmode_blocking)
        vrep.simxGetPingTime(self.venv.cid)

    def get_position(self):
            return np.asarray(vrep.simxGetObjectPosition(self.venv.cid, self.robot['body'], -1,
                                                     vrep.simx_opmode_blocking)[1][:2])
    def get_orientation(self, axis=0):
        return vrep.simxGetObjectOrientation(self.venv.cid, self.robot['body'], -1,
                                                     vrep.simx_opmode_blocking)[1][axis]

    def reset(self):
        self.venv.stop_simulation()
        self.position = self.get_position()
        self.set_random_target()
        self.distance = np.sqrt(np.sum(np.power(self.dest - self.position, 2)))
        self.venv.start_blocking_simulation()
        vrep.simxGetPingTime(self.venv.cid)

    def set_random_target(self):
        while True:
            self.dest = np.random.rand(2) * 10 - 5
            if np.sqrt(np.sum(np.power(self.dest - self.position, 2))) > 2.:
                break

    def step(self, speed):
        speed = speed.flatten()
        self.previous_speed = speed
        self.send_data(speed)
        self.venv.step_blocking_simulation()
        data = self.read_data(step=True)
        done = self.distance < 1.0
        reward = (self.prev_distance - self.distance)*100.
        # if np.any(data[:5] < 0.2):
        #     reward -= (0.15-np.min(data[:5])) * 50.
        # if np.all(speed < 0):
        #     reward -= 2
        # if 0. < reward < 1.:
        #     reward = np.power(reward, 2)
        have_to_reset = abs(self.get_orientation()) > 0.01
        return data, reward, done, have_to_reset

    def done(self):
        self.venv.stop_simulation()
        vrep.simxFinish(self.venv.cid)


if __name__ == "__main__":
    robot = Q_learning(headless=False)
    while True:
        print(robot.get_orientation())
        d, r, m, h = robot.step(np.array([1, 0.2]))


