import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import numpy as np
from collections import deque
import math
import tensorflow as tf
from networks import Networks
from field_env import Environment
import threading
from keras import backend as K

scene_path = os.path.dirname(os.getcwd()) + '/scenes/field.ttt'
POSSIBLE_ACTIONS = np.array([[-1., 1.], [1., -1.], [1., 1.]])
episode = 0
TAU = 0.001




class C51Agent:
    def __init__(self, image_size, action_size, num_atoms, l_r, memory):
        # get size of state and action
        self.image_size = image_size
        self.action_size = action_size
        self.memory = memory

        self.max_memory = 25000
        self.env = Environment(headless=False,scene_path=scene_path)
        # these is hyper parameters for the DQN
        self.gamma = 0.997
        self.learning_rate = l_r          # 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0         # 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32              # 32
        self.observe = 5000
        self.explore = 50000
        self.frame_per_action = 1       #4
        self.update_target_freq = 1000          #3000
        self.timestep_per_train = 100  # Number of timesteps between training interval       # 100
        # Initialize Atoms
        self.num_atoms = num_atoms  # 51 for C51
        self.v_max = 20  # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        self.v_min = 0
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]
        # Models for value distribution
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.stats_window_size = 500  # window size for computing rolling statistics

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, image):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            # print("----------Random Action----------")
            action = random.randrange(self.action_size)
        else:
            action = self.get_optimal_action(image)

        return action

    def get_optimal_action(self, image):
        """Get optimal action for a state
        """
        pred = self.model.predict(image[np.newaxis, :])
        z = pred
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        # Pick action with the biggest Q value
        action = np.argmax(q)
        return action

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, image, action, r, next_image, done, global_step):
        self.memory.append((image, action, r, next_image, done, global_step))
        if self.epsilon > self.final_epsilon and global_step > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        # Update the target model to be same with model
        if global_step % self.update_target_freq == 0:
            self.update_target_model()

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):

        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        image_inputs = np.zeros(((num_samples,) + self.image_size))
        next_images = np.zeros(((num_samples,) + self.image_size))
        m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(action_size)]
        action, reward, done = [], [], []

        for i in range(num_samples):
            image_inputs[i, :, :, :] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            next_images[i, :, :, :] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        prediction = self.model.predict(next_images)
        prediction_ = self.target_model.predict(next_images)
        z = prediction
        z_ = prediction_

        # Get Optimal Actions for the next states (from distribution z)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)  # length (num_atoms x num_actions)
        q = q.reshape((num_samples, action_size), order='F')
        optimal_action_idxs = np.argmax(q, axis=1)
        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):
            if done[i]:  # Terminal State
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, reward[i]))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action[i]][i][int(m_l)] += (m_u - bj)
                m_prob[action[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
                    m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)
        loss = self.model.fit(image_inputs, m_prob, batch_size=self.batch_size, epochs=2, verbose=0)
        return loss.history['loss']

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

    def work(self, weights_path, worker_number):
        global episode
        global_step = 0
        while episode < MAX_EP:
            self.env.reset()
            image = self.env.read_data()
            ep_r = 0
            for step in range(MAX_EP_STEP):
                global_step += 1
                a = self.get_action(image)
                next_image, r, done = self.env.step(POSSIBLE_ACTIONS[a])  # make step in environment
                if not done:
                    done = True if step == MAX_EP_STEP - 1 else False
                ep_r += r
                self.replay_memory(image, a, r, next_image, done, global_step)

                # train
                if global_step > self.observe and global_step % self.timestep_per_train == 0:
                    loss = self.train_replay()
                image = next_image
                if global_step % 10000 == 0:
                    print("Now we save model")
                    self.model.save_weights(weights_path, overwrite=True)
                if done:
                    print("Episode: {0} Reward: {1} Worker {2}".format(episode, ep_r, worker_number))
                    break
            episode += 1


if __name__ == "__main__":
    weights_name = "c51_ddqn"        ###
    workers = 4
    # Parameters
    action_size = 3
    MAX_EP = 1000000
    MAX_EP_STEP = 2500
    learning_rate = 0.0001
    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    img_rows, img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 4  # We stack 4 frames   # I use 1 RGB frame
    # C51
    num_atoms = 51
    image_size = (img_rows, img_cols, img_channels)
    target_model = Networks.value_distribution_network(sess, image_size, num_atoms, action_size, learning_rate)
    replay_buffer = deque()
    for w in range(workers):
        agent = C51Agent(image_size, action_size, num_atoms, learning_rate, replay_buffer)
        agent.model = Networks.value_distribution_network(sess, image_size, num_atoms, action_size, agent.learning_rate)
        agent.target_model = target_model
        worker_weights_name = weights_name + '_worker_' + str(w) + '.h5'
        if worker_weights_name in os.listdir(os.getcwd()):
            agent.load_model(worker_weights_name)
        worker_work = lambda: agent.work(worker_weights_name, str(w))
        t = threading.Thread(target=(worker_work))
        t.start()



