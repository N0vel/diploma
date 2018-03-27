import sys
import os
import numpy as np
from vrep import vrep
import math
import random
import subprocess as sp
import types
from inspect import getfullargspec


list_of_instances = []  
class instance():
    def __init__(self, args):
        self.args = args
        list_of_instances.append(self)

    def start(self):
        print('(instance) starting...')
        try:
            self.inst = sp.Popen(self.args)
        except EnvironmentError:
            print('(instance) Error: cannot find executable at', self.args[0])
            raise

        return self

    def isAlive(self):
        return True if self.inst.poll() is None else False

    def end(self):
        print('(instance) terminating...')
        if self.isAlive():
            self.inst.terminate()
            retcode = self.inst.wait()
        else:
            retcode = self.inst.returncode
        print('(instance) retcode:', retcode)
        return self

# взаимодействие с v-rep
class Q_learning():
    def __init__(self, headless=False, port_num=None, dir_vrep="C:/Program Files/V-REP3/V-REP_PRO_EDU/", scene_path=''):
        self.possible_actions = np.linspace(-10, 10, 41)
        if port_num is None:
            self.port_num = int(random.random() * 1000 + 19999)
        if dir_vrep == '':
            print('(vrepper) trying to find V-REP executable in your PATH')
            import distutils.spawn as dsp
            path_vrep = dsp.find_executable('vrep.sh')  # fix for linux
            if path_vrep == None:
                path_vrep = dsp.find_executable('vrep')
        else:
            path_vrep = dir_vrep + 'vrep'
        print('(vrepper) path to your V-REP executable is:', path_vrep)
        args = [path_vrep, '-gREMOTEAPISERVERSERVICE_' + str(self.port_num) + '_FALSE_TRUE']
        if headless:
            args.append('-h')
        self.instance = instance(args)
        self.cid = -1
        vrep_methods = [a for a in dir(vrep) if
                        not a.startswith('__') and isinstance(getattr(vrep, a), types.FunctionType)]
        def assign_from_vrep_to_self(name):
            wrapee = getattr(vrep, name)
            arg0 = getfullargspec(wrapee)[0][0]
            if arg0 == 'clientID':
                def func(*args, **kwargs):
                    return wrapee(self.cid, *args, **kwargs)
            else:
                def func(*args, **kwargs):
                    return wrapee(*args, **kwargs)
            setattr(self, name, func)
        for name in vrep_methods:
            assign_from_vrep_to_self(name)
        self.start()
        self.check_ret(self.simxLoadScene(scene_path, 0, vrep.simx_opmode_blocking))
        self.sensors = ['Vision_sensor']
        self.motors = ['motor_left_1', 'motor_left_2', 'motor_left_3', 'motor_right_1', 'motor_right_2',
                       'motor_right_3']
        self.robot = self.get_object_handles()
        self.position = self.get_position()
        self.set_random_target()
        self.prev_distance = np.sqrt(np.sum(np.power(self.dest - self.position, 2)))
        self.distance = np.copy(self.prev_distance)
        self.prev_angle = self.angle_between(self.position, self.dest, self.get_orientation(axis=2))
        self.angle = np.copy(self.prev_angle)
        self.data = np.zeros(2)
        self.start_simulation(True)

    def start_simulation(self, is_sync):
        # IMPORTANT
        # you should poll the server state to make sure
        # the simulation completely stops before starting a new one
        while True:
            # poll the useless signal (to receive a message from server)
            self.check_ret(self.simxGetIntegerSignal(
                'asdf', vrep.simx_opmode_blocking))

            # check server state (within the received message)
            e = self.simxGetInMessageInfo(
                vrep.simx_headeroffset_server_state)

            # check bit0
            not_stopped = e[1] & 1

            if not not_stopped:
                break

        # enter sync mode
        self.check_ret(self.simxSynchronous(is_sync))
        self.check_ret(self.simxStartSimulation(vrep.simx_opmode_blocking))
        self.sim_running = True


    def start(self):
        print('(vrepper)starting an instance of V-REP...')
        self.instance.start()

        # try to connect to V-REP instance via socket
        retries = 0
        while True:
            print('(vrepper)trying to connect to server on port', self.port_num, 'retry:', retries)
            # vrep.simxFinish(-1)  # just in case, close all opened connections
            self.cid = self.simxStart(
                '127.0.0.1', self.port_num,
                waitUntilConnected=True,
                doNotReconnectOnceDisconnected=True,
                timeOutInMs=10000,
                commThreadCycleInMs=0)  # Connect to V-REP

            if self.cid != -1:
                print('(vrepper)Connected to remote API server!')
                break
            else:
                retries += 1
                if retries > 15:
                    self.end()
                    raise RuntimeError('(vrepper)Unable to connect to V-REP after 15 retries.')

        # Now try to retrieve data in a blocking fashion (i.e. a service call):
        objs, = self.check_ret(self.simxGetObjects(vrep.sim_handle_all,vrep.simx_opmode_blocking))
        print('(vrepper)Number of objects in the scene: ', len(objs))
        # Now send some data to V-REP in a non-blocking fashion:
        self.simxAddStatusbarMessage(
            '(vrepper)Hello V-REP!',
            vrep.simx_opmode_oneshot)
        # setup a useless signal
        self.simxSetIntegerSignal('asdf', 1, vrep.simx_opmode_blocking)

    def check_ret(self, ret_tuple, ignore_one=False):
        istuple = isinstance(ret_tuple, tuple)
        if not istuple:
            ret = ret_tuple
        else:
            ret = ret_tuple[0]

        if (not ignore_one and ret != vrep.simx_return_ok) or (ignore_one and ret > 1):
            raise RuntimeError('retcode(' + str(ret) + ') not OK, API call failed. Check the parameters!')

        return ret_tuple[1:] if istuple else None

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
        vrep_parts = ['Caterpillar_respondable', 'Vision_sensor',
                      'dynamicLeftJoint1', 'dynamicLeftJoint2', 'dynamicLeftJoint3',
                      'dynamicRightJoint1', 'dynamicRightJoint2', 'dynamicRightJoint3']
        for i, part in enumerate(['body'] + self.sensors + self.motors):
            robot[part] = vrep.simxGetObjectHandle(self.cid, vrep_parts[i], vrep.simx_opmode_blocking)[1]
        return robot

    def read_data(self, step):
        res, resolution, image = vrep.simxGetVisionSensorImage(self.cid, self.robot[self.sensors[0]], 0, vrep.simx_opmode_blocking)
        self.image = np.reshape(np.array(image, dtype='float32')+128, (64, 64, 3)) / 255.
        self.position = self.get_position()
        if step:
            self.prev_distance = np.copy(self.distance)
            self.prev_angle = np.copy(self.angle)
            self.data[-2] = self.distance = np.sqrt(np.sum(np.power(self.dest - self.position, 2)))
            self.data[-1] = self.angle = self.angle_between(self.position, self.dest, self.get_orientation(axis=2))
        else:
            self.data[-2] = np.sqrt(np.sum(np.power(self.dest - self.position, 2)))
            self.data[-1] = self.angle_between(self.position, self.dest, self.get_orientation(axis=2))
        return self.data, self.image

    def send_data(self, speed):
        # speed = -10. * speed
        for i, motor in enumerate(self.motors):
            vrep.simxSetJointTargetVelocity(self.cid, self.robot[motor], speed[i // 3], vrep.simx_opmode_blocking)

    def get_position(self):
            return np.asarray(vrep.simxGetObjectPosition(self.cid, self.robot['body'], -1,
                                                     vrep.simx_opmode_blocking)[1][:2])
    def get_orientation(self, axis=0):
        return vrep.simxGetObjectOrientation(self.cid, self.robot['body'], -1,
                                                     vrep.simx_opmode_blocking)[1][axis]

    def reset(self):
        self.check_ret(self.simxStopSimulation(vrep.simx_opmode_oneshot), ignore_one=True)
        self.position = self.get_position()
        self.set_random_target()
        self.distance = np.sqrt(np.sum(np.power(self.dest - self.position, 2)))
        self.start_simulation(True)
        vrep.simxGetPingTime(self.cid)

    def set_random_target(self):
        while True:
            self.dest = np.random.rand(2) * 10 - 5
            if np.sqrt(np.sum(np.power(self.dest - self.position, 2))) > 2.:
                break

    def step(self, speed):
        # speed = speed.flatten()
        speed = [self.possible_actions[speed[0]], self.possible_actions[speed[1]]]
        self.send_data(speed)
        self.check_ret(self.simxSynchronousTrigger())
        self.read_data(step=True)
        done = self.distance < 1.0
        reward = (self.prev_distance - self.distance) * 10.
        if np.all(np.array(speed) < 0):
            reward -= 0.1
        return self.data, self.image, reward, done

    def done(self):
        self.check_ret(self.simxStopSimulation(vrep.simx_opmode_oneshot), ignore_one=True)
        vrep.simxFinish(self.cid)


if __name__ == "__main__":
    scene_path = os.getcwd() + '/scenes/diploma.ttt'
    robot_1 = Q_learning(headless=False, scene_path=scene_path)
    robot_2 = Q_learning(headless=False, scene_path=scene_path)
    robot_3 = Q_learning(headless=False, scene_path=scene_path)
    robot_4 = Q_learning(headless=False, scene_path=scene_path)
    while True:
        d, img, r, w = robot_1.step(np.array([1, 0.2]))
        d, img, r, w = robot_2.step(np.array([0.2, 1]))
        d, img, r, w = robot_3.step(np.array([1, 0.2]))
        d, img, r, w = robot_4.step(np.array([0.2, 1]))


