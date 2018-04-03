
import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from random import choice
from time import sleep
from time import time
import os
from field_env import Environment
import random
# YOU HAVE CHANGED LR
scene_path = os.path.dirname(os.getcwd()) + '/scenes/field.ttt'
POSSIBLE_ACTIONS = np.linspace(-1, 1, 41)
EPSILON = 1.0
EPSILON_STEP = 0.000001

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.image_prep = tf.reshape(self.inputs, shape=[-1, 64, 64, 4])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.image_prep, num_outputs=32,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1, num_outputs=64,
                                     kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2), 512, activation_fn=tf.nn.elu)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.image_prep)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            self.policy_1 = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.policy_2 = slim.fully_connected(rnn_out, a_size,
                                                 activation_fn=tf.nn.softmax,
                                                 weights_initializer=normalized_columns_initializer(0.01),
                                                 biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions_1 = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot_1 = tf.one_hot(self.actions_1, a_size, dtype=tf.float32)
                self.actions_2= tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot_2 = tf.one_hot(self.actions_2, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs_1 = tf.reduce_sum(self.policy_1 * self.actions_onehot_1, [1])
                self.responsible_outputs_2 = tf.reduce_sum(self.policy_2 * self.actions_onehot_2, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy_1 = - tf.reduce_sum(self.policy_1 * tf.log(self.policy_1))
                self.entropy_2 = - tf.reduce_sum(self.policy_2 * tf.log(self.policy_2))
                self.policy_loss_1 = -tf.reduce_sum(tf.log(self.responsible_outputs_1) * self.advantages)
                self.policy_loss_2 = -tf.reduce_sum(tf.log(self.responsible_outputs_2) * self.advantages)
                self.loss = (self.value_loss + self.policy_loss_1 + self.policy_loss_2
                             - self.entropy_1 * 0.01 - self.entropy_2 * 0.01) / 2.

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        # The Below code is related to setting up the Doom environment

        self.actions_1 = self.actions_1 = np.identity(a_size, dtype=bool).tolist()
        self.actions_2 = self.actions_2 = np.identity(a_size, dtype=bool).tolist()
        self.env = Environment(headless=False, scene_path=scene_path)

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions_1 = rollout[:,1]
        actions_2 = rollout[:,6]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions_1: actions_1,
                     self.local_AC.actions_2: actions_2,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}
        v_l, p_l1, p_l2, e_l1, e_l2, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
                                                                     self.local_AC.policy_loss_1,
                                                                     self.local_AC.policy_loss_2,
                                                                     self.local_AC.entropy_1,
                                                                     self.local_AC.entropy_2,
                                                                     self.local_AC.grad_norms,
                                                                     self.local_AC.var_norms,
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],
                                                                    feed_dict=feed_dict)
        return v_l / len(rollout), p_l1 / len(rollout), p_l2 / len(rollout), e_l1 / len(rollout), \
               e_l2 / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        global EPSILON
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                self.env.reset()
                s = self.env.read_data()
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                while episode_step_count < max_episode_length:
                    # Take an action using probabilities from policy network output.
                    a_dist1, a_dist2, v, rnn_state = sess.run(
                        [self.local_AC.policy_1, self.local_AC.policy_2, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})
                    if random.random() > EPSILON:
                        a1 = np.argmax(a_dist1)
                        a2 = np.argmax(a_dist2)
                        # a1 = np.random.choice(a_dist1[0], p=a_dist1[0])
                        # a1 = np.argmax(a_dist1 == a1)
                        # a2 = np.random.choice(a_dist2[0], p=a_dist2[0])
                        # a2 = np.argmax(a_dist2 == a2)
                    else:
                        a1, a2 = random.choice(range(35, 41)), random.choice(range(35, 41))
                    EPSILON -= EPSILON_STEP
                    s1, r, d = self.env.step(np.array([POSSIBLE_ACTIONS[a1], POSSIBLE_ACTIONS[a2]]))
                    episode_buffer.append([s, a1, r, s1, d, v[0, 0], a2])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: [s],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                        v_l, p_l1, p_l2, e_l1, e_l2, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                print('{0}: {1} | Epsilon: {2}'.format(self.name, episode_reward, EPSILON))
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l1, p_l2, e_l1, e_l2, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 100 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy_1 Loss', simple_value=float(p_l1))
                    summary.value.add(tag='Losses/Policy_2 Loss', simple_value=float(p_l2))
                    summary.value.add(tag='Losses/Entropy_1', simple_value=float(e_l1))
                    summary.value.add(tag='Losses/Entropy_2', simple_value=float(e_l2))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1


if __name__=='__main__':
    max_episode_length = 2500
    gamma = .99 # discount rate for advantage estimation and reward discounting
    s_size = 64*64*4 # Observations are 4 history frames of 64 * 64 * 4
    a_size = 41
    load_model = False
    model_path = './model'

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')
    with tf.device("/gpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)        ###################################
        master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
        num_workers = 4
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(i, s_size, a_size, trainer, model_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)