import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from q_learning import Q_learning


# PARAMETERS
OUTPUT_GRAPH = True  # safe logs
LOG_DIR = './log'  # savelocation for logs
N_WORKERS = 4
MAX_EP_STEP = 5000  # maxumum number of steps per episode
MAX_GLOBAL_EP = 100000  # total number of episodes
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 1  # sets how often the global net is updated  # 10
GAMMA = 0.99  # discount factor                                    #0.9
# ENTROPY_BETA = 0.01  # entropy factor                                #0.01
ENTROPY_BETA = 0.1
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
# set environment
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3
data_size = 2
N_A = 2  # number of actions
A_BOUND = [-1., 1.]  # action bounds




class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.image = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], 'image')
                self.d_input = tf.placeholder(tf.float32, [None, data_size], 'data')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.image = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], 'image')
                self.d_input = tf.placeholder(tf.float32, [None, data_size], 'data')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('critic'):   # only critic controls the rnn update
            cell_size = 256
            conv1 = tf.layers.conv2d(
                inputs=self.image,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu6)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu6)
            flat = tf.layers.flatten(conv2)
            dense_1 = tf.layers.dense(self.d_input, units=256, activation=tf.nn.tanh, kernel_initializer=w_init)
            dense_2 = tf.layers.dense(dense_1, units=256, activation=tf.nn.tanh, kernel_initializer=w_init)
            concat = tf.concat([flat, dense_2], axis=1)
            gru_in = tf.expand_dims(concat, axis=1, name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            gru_cell = tf.contrib.rnn.GRUCell(cell_size)
            self.init_state = gru_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(cell=gru_cell, inputs=gru_in, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation
            l_c = tf.layers.dense(cell_out, 256, activation=tf.nn.tanh, kernel_initializer=w_init, name='lc')      #consider activation
            v = tf.layers.dense(l_c, 1, activation=None, kernel_initializer=w_init, name='v')  # state value

        with tf.variable_scope('actor'):  # state representation is based on critic
            l_a_1 = tf.layers.dense(cell_out, 256, tf.nn.tanh, kernel_initializer=w_init, name='la1')
            l_a_2 = tf.layers.dense(l_a_1, 256, tf.nn.tanh, kernel_initializer=w_init, name='la2')
            mu = tf.layers.dense(l_a_2, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a_2, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, data, image, cell_state):  # run by a local
        data = data[np.newaxis, :]
        image = image[np.newaxis, :]
        a, cell_state = SESS.run([self.A, self.final_state], {self.d_input: data, self.image: image, self.init_state: cell_state})
        return a[0], cell_state


# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, name, globalAC):
        scene_path = os.path.dirname(os.getcwd()) + '/scenes/diploma.ttt'
        self.env = Q_learning(headless=True, scene_path=scene_path)
        self.name = name
        self.AC = ACNet(name, globalAC)  # create ACNet for each worker

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_image, buffer_data, buffer_a, buffer_r = [], [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            self.env.reset()
            data, image = self.env.read_data(step=False)
            ep_r = 0
            rnn_state = SESS.run(self.AC.init_state)  # zero rnn state at beginning
            keep_state = rnn_state.copy()  # keep rnn state for updating global net
            for ep_t in range(MAX_EP_STEP):
                a, rnn_state_ = self.AC.choose_action(data, image, rnn_state)  # get the action and next rnn state
                next_data, next_image, r, done = self.env.step(a)  # make step in environment
                if not done:
                    done = True if ep_t == MAX_EP_STEP - 1 else False
                ep_r += r
                buffer_image.append(image[np.newaxis, :])
                buffer_data.append(data[np.newaxis, :])
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.image: next_image[np.newaxis, :],
                                                    self.AC.d_input: next_data[np.newaxis, :],
                                                    self.AC.init_state: rnn_state_})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_image, buffer_data, buffer_a, buffer_v_target = np.vstack(buffer_image), np.vstack(buffer_data),\
                                                                           np.vstack(buffer_a), np.vstack(buffer_v_target)

                    feed_dict = {
                        self.AC.image: buffer_image,
                        self.AC.d_input: buffer_data,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.init_state: keep_state,
                    }

                    self.AC.update_global(feed_dict)
                    buffer_image, buffer_data, buffer_a, buffer_r = [], [], [], []
                    self.AC.pull_global()
                    keep_state = rnn_state_.copy()   # replace the keep_state as the new initial rnn state_

                image, data = next_image, next_data
                rnn_state = rnn_state_  # renew rnn state
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    SESS = tf.Session()

    OPT_A = tf.train.AdamOptimizer(LR_A, name='AdamA')
    OPT_C = tf.train.AdamOptimizer(LR_C, name='AdamC')
    GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
    workers = []
    # Create worker
    for i in range(N_WORKERS):
        i_name = 'W_%i' % i  # worker name
        workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()