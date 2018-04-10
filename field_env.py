import sys
import os
import numpy as np
from vrep import vrep
import math
import random
import subprocess as sp
import types
from inspect import getfullargspec
import matplotlib.pyplot as plt
import distutils as dsp


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
class Environment():
    def __init__(self, headless=False, port_num=None, dir_vrep="C:/Program Files/V-REP3/V-REP_PRO_EDU/", scene_path='/scenes/field.ttt'):
        if port_num is None:
            self.port_num = int(random.random() * 1000 + 19999)
        if dir_vrep == '':
            print('Trying to find V-REP executable in your PATH')
            import distutils.spawn as dsp
            path_vrep = dsp.find_executable('vrep.sh')  # fix for linux
            if path_vrep == None:
                path_vrep = dsp.find_executable('vrep')
        else:
            path_vrep = dir_vrep + 'vrep'
        print('Path to your V-REP executable is:', path_vrep)
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
        self.n_targets = 10
        self.check_ret(self.simxLoadScene(scene_path, 0, vrep.simx_opmode_blocking))
        self.sensors = ['Vision_sensor']
        self.motors = ['motor_left_1', 'motor_left_2', 'motor_left_3', 'motor_right_1', 'motor_right_2',
                       'motor_right_3']
        self.g_t_names = ['Sphere' + str(i) for i in range(self.n_targets)]
        self.b_t_names = ['Cuboid' + str(i) for i in range(self.n_targets)]
        self.robot = self.get_object_handles()
        self.set_random_targets()
        self.get_position()
        self.visited_good = []
        self.visited_bad = []
        # self.image_buf = np.zeros((4, 64, 64))
        self.image_buf = np.zeros((64, 64, 4))
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
            print('Trying to connect to server on port', self.port_num, 'retry:', retries)
            # vrep.simxFinish(-1)  # just in case, close all opened connections
            self.cid = self.simxStart(
                '127.0.0.1', self.port_num,
                waitUntilConnected=True,
                doNotReconnectOnceDisconnected=True,
                timeOutInMs=10000,
                commThreadCycleInMs=0)  # Connect to V-REP

            if self.cid != -1:
                print('Connected to remote API server!')
                break
            else:
                retries += 1
                if retries > 15:
                    self.end()
                    raise RuntimeError('Unable to connect to V-REP after 15 retries.')

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

    def get_object_handles(self):
        robot = {}
        vrep_parts = ['Caterpillar_respondable', 'Vision_sensor',
                      'dynamicLeftJoint1', 'dynamicLeftJoint2', 'dynamicLeftJoint3',
                      'dynamicRightJoint1', 'dynamicRightJoint2', 'dynamicRightJoint3']
        for i, part in enumerate(['body'] + self.sensors + self.motors):
            robot[part] = vrep.simxGetObjectHandle(self.cid, vrep_parts[i], vrep.simx_opmode_blocking)[1]
        self.g_t_handles = [vrep.simxGetObjectHandle(self.cid, self.g_t_names[i], vrep.simx_opmode_blocking)[1] for i in range(self.n_targets)]
        self.b_t_handles = [vrep.simxGetObjectHandle(self.cid, self.b_t_names[i], vrep.simx_opmode_blocking)[1] for i in range(self.n_targets)]
        return robot

    def get_position(self):
        self.position = np.asarray(vrep.simxGetObjectPosition(self.cid, self.robot['body'], -1,
                                                     vrep.simx_opmode_blocking)[1][:2])

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    # def extend_history_buffer(self, image):
    #     self.image_buf[0, :, :], self.image_buf[1, :, :], self.image_buf[2, :, :],\
    #         self.image_buf[3, :, :] = self.image_buf[1, :, :], self.image_buf[2, :, :], self.image_buf[3, :, :], image

    def extend_history_buffer(self, image):
        self.image_buf[:, :, 0], self.image_buf[:, :, 1], self.image_buf[:, :, 2], self.image_buf[:, :, 3] = \
            self.image_buf[:, :, 1], self.image_buf[:, :, 2], self.image_buf[:, :, 3], image

    def read_data(self):
        res, resolution, image = vrep.simxGetVisionSensorImage(self.cid, self.robot[self.sensors[0]], 0, vrep.simx_opmode_blocking)
        image = self.rgb2gray(np.reshape(np.array(image, dtype='float32')+128, (64, 64, 3))) / 255.
        self.extend_history_buffer(image)
        # return self.image_buf.reshape(1, 64, 64, 4)
        return self.image_buf

    def send_data(self, speed):
        speed = speed * -10
        for i, motor in enumerate(self.motors):
            vrep.simxSetJointTargetVelocity(self.cid, self.robot[motor], speed[i // 3], vrep.simx_opmode_blocking)

    def distance(self, p1, p2):
        return np.sqrt(np.sum(np.power(p1 - p2, 2)))

    def reset(self):
        self.check_ret(self.simxStopSimulation(vrep.simx_opmode_oneshot), ignore_one=True)
        self.get_position()
        self.clear_targets()
        self.visited_good = []
        self.visited_bad = []
        self.set_random_targets()
        self.start_simulation(True)
        vrep.simxGetPingTime(self.cid)

    def clear_targets(self):
        for handle in self.g_t_handles:
            vrep.simxSetObjectPosition(self.cid, handle, relativeToObjectHandle=-1,
                                           position=(0., 0., -1.),
                                           operationMode=vrep.simx_opmode_blocking)
        for handle in self.b_t_handles:
            vrep.simxSetObjectPosition(self.cid, handle, relativeToObjectHandle=-1,
                                       position=(0., 0., -1.),
                                       operationMode=vrep.simx_opmode_blocking)
    def set_random_targets(self):
        self.good_targets = list((np.random.rand(self.n_targets*2).reshape((self.n_targets, 2))- 0.5)*9)
        self.bad_targets = list((np.random.rand(self.n_targets*2).reshape((self.n_targets, 2)) - 0.5)*9)
        for i,handle in enumerate(self.g_t_handles):
            vrep.simxSetObjectPosition(self.cid, handle, relativeToObjectHandle=-1,
                                       position=(self.good_targets[i][0], self.good_targets[i][1], 0.125),
                                       operationMode=vrep.simx_opmode_blocking)
        for i,handle in enumerate(self.b_t_handles):
            vrep.simxSetObjectPosition(self.cid, handle, relativeToObjectHandle=-1,
                                       position=(self.bad_targets[i][0], self.bad_targets[i][1], 0.125),
                                       operationMode=vrep.simx_opmode_blocking)

    def step(self, speed):
        speed = speed.flatten()
        self.send_data(speed)
        self.check_ret(self.simxSynchronousTrigger())
        image = self.read_data()
        self.get_position()
        reward = 0
        for i, target in enumerate(self.good_targets):
            if self.distance(self.position, target) < 0.5 and i not in self.visited_good:
                self.visited_good.append(i)
                vrep.simxSetObjectPosition(self.cid, self.g_t_handles[i], relativeToObjectHandle=-1,
                                           position=(0., 0., -1.),
                                           operationMode=vrep.simx_opmode_blocking)
                reward += 1
        for i, target in enumerate(self.bad_targets):
            if self.distance(self.position, target) < 0.5 and i not in self.visited_bad:
                self.visited_bad.append(i)
                vrep.simxSetObjectPosition(self.cid, self.b_t_handles[i], relativeToObjectHandle=-1,
                                           position=(0., 0., -1.),
                                           operationMode=vrep.simx_opmode_blocking)
                reward -= -1

        done = len(self.visited_good) == self.n_targets
        if vrep.simxGetObjectPosition(self.cid, self.robot['body'], -1, vrep.simx_opmode_blocking)[1][2] < -1.:
            done = True
        return image, reward, done

    def done(self):
        self.check_ret(self.simxStopSimulation(vrep.simx_opmode_oneshot), ignore_one=True)
        vrep.simxFinish(self.cid)


if __name__ == "__main__":
    scene_path = os.getcwd() + '/scenes/field.ttt'
    robot_1 = Environment(headless=False, scene_path=scene_path)
    while True:
        robot_1.send_data(np.array([0.2, 0.5]))
        robot_1.check_ret(vrep.simxSynchronousTrigger(robot_1.cid))
        image = robot_1.read_data()
        plt.imshow(image)
        plt.show()
    # robot_2 = Environment(headless=False, scene_path=scene_path)
    # robot_3 = Environment(headless=False, scene_path=scene_path)
    # robot_4 = Environment(headless=False, scene_path=scene_path)
    # while True:
    #     img, r, d = robot_1.step(np.array([1, 0.2]))
        # img, r, d = robot_2.step(np.array([0.2, 1]))
        # img, r, d = robot_3.step(np.array([1, 0.2]))
        # img, r, d = robot_4.step(np.array([0.2, 1]))


