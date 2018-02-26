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
# N_WORKERS = multiprocessing.cpu_count()  # number of workers
N_WORKERS = 1
MAX_EP_STEP = 5000  # maxumum number of steps per episode
MAX_GLOBAL_EP = 100000  # total number of episodes
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 1  # sets how often the global net is updated  # 10
GAMMA = 0.99  # discount factor                                    #0.9
ENTROPY_BETA = 0.2  # entropy factor                                #0.01
LR_A = 0.0005  # learning rate for actor
LR_C = 0.001  # learning rate for critic

# set environment
N_S = 9  # number of states
N_A = 2  # number of actions
A_BOUND = [-1., 1.]  # action bounds

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Network for the Actor Critic
class ACNet(object):
    def __init__(self, scope, sess, globalAC=None):
        self.sess = sess
        self.actor_optimizer = tf.train.AdamOptimizer(LR_A, name='RMSPropA')  # optimizer for the actor
        self.critic_optimizer = tf.train.AdamOptimizer(LR_C, name='RMSPropC')  # optimizer for the critic

        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_params, self.c_params = self._build_net(scope)[-2:]  # parameters of actor and critic net
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')  # action
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # v_target value

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(
                    scope)  # get mu and sigma of estimated action from neural net

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0],
                                              A_BOUND[1])  # sample a action from distribution
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss,
                                                self.a_params)  # calculate gradients for the network weights
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):  # update local and global network weights
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):  # neural network structure of the actor and critic
        w_init = tf.glorot_normal_initializer()
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 512, tf.nn.tanh, kernel_initializer=w_init, name='la')
            a_d = tf.layers.dropout(inputs=l_a, rate=0.25)

            l_a1 = tf.layers.dense(a_d, 512, tf.nn.tanh, kernel_initializer=w_init, name='l_a1')
            a_d1 = tf.layers.dropout(inputs=l_a1, rate=0.25)

            l_a2 = tf.layers.dense(a_d1, 512, tf.nn.tanh, kernel_initializer=w_init, name='l_a2')
            a_d2 = tf.layers.dropout(inputs=l_a2, rate=0.25)

            l_a3 = tf.layers.dense(a_d2, 512, tf.nn.tanh, kernel_initializer=w_init, name='l_a3')
            a_d3 = tf.layers.dropout(inputs=l_a3, rate=0.25)

            mu = tf.layers.dense(a_d3, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')  # estimated action value
            sigma = tf.layers.dense(a_d3, N_A, tf.nn.softplus, kernel_initializer=w_init,
                                    name='sigma')  # estimated variance
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 512, tf.nn.tanh, kernel_initializer=w_init, name='lc')
            c_d = tf.layers.dropout(inputs=l_c, rate=0.25)

            l_c1 = tf.layers.dense(c_d, 512, tf.nn.tanh, kernel_initializer=w_init, name='l_c1')
            c_d1 = tf.layers.dropout(inputs=l_c1, rate=0.25)

            l_c2 = tf.layers.dense(c_d1, 512, tf.nn.tanh, kernel_initializer=w_init, name='l_c2')
            c_d2 = tf.layers.dropout(inputs=l_c2, rate=0.25)

            l_c3 = tf.layers.dense(c_d2, 512, tf.nn.tanh, kernel_initializer=w_init, name='l_c3')
            c_d3 = tf.layers.dropout(inputs=l_c3, rate=0.25)

            v = tf.layers.dense(c_d3, 1, kernel_initializer=w_init, name='v')  # estimated value for state
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.s: s})[0]


# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, name, globalAC, sess):
        self.env = Q_learning(headless=False)
        self.name = name
        self.AC = ACNet(name, sess, globalAC)  # create ACNet for each worker
        self.sess = sess

    def work(self):
        global global_rewards, global_episodes
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not coord.should_stop() and global_episodes < MAX_GLOBAL_EP:
            self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                s = self.env.read_data(step=False)
                a = self.AC.choose_action(s)  # estimate stochastic action based on policy
                s_, r, done, info = self.env.step(a)  # make step in environment
                if not done:
                    done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                # save actions, states and rewards in buffer
                buffer_s.append(s)
                buffer_a.append(a)
                # buffer_r.append((r + 8) / 8)  # normalize reward    ##################################
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done or info:  # update global and assign to local net
                    if done or info:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()  # get global parameters to local ACNet

                s = s_
                total_step += 1
                if done or info:
                    if len(global_rewards) < 5:  # record running episode reward
                        global_rewards.append(ep_r)
                    else:
                        global_rewards.append(ep_r)
                        global_rewards[-1] = (np.mean(global_rewards[-5:]))  # smoothing
                    print(
                        self.name,
                        "Ep:", global_episodes,
                        "| Ep_r: %i" % global_rewards[-1],
                    )
                    global_episodes += 1
                    break


if __name__ == "__main__":
    global_rewards = []
    global_episodes = 0

    sess = tf.Session()

    with tf.device("/cpu:0"):
        global_ac = ACNet(GLOBAL_NET_SCOPE, sess)  # we only need its params
        workers = []
        # Create workersgit reset --hard tags/v2.0
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, global_ac, sess))

    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:  # write log file
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)

    worker_threads = []
    for worker in workers:  # start workers
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)  # wait for termination of workers

    plt.plot(np.arange(len(global_rewards)), global_rewards)  # plot rewards
    plt.xlabel('step')
    plt.ylabel('total moving reward')
    plt.show()