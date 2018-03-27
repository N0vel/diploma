import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import numpy as np
from collections import deque
import math
from keras import backend as K
import tensorflow as tf

from networks import Networks
from q_learning import Q_learning

class C51Agent:
    def __init__(self, image_size, data_size, action_size, num_atoms):

        # get size of state and action
        self.image_size = image_size
        self.data_size = data_size
        self.action_size = action_size

        # these is hyper parameters for the DQN
        self.gamma = 0.99
        self.learning_rate = 0.001          # 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0         # 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 8                # 32
        self.observe = 5000
        self.explore = 50000
        self.frame_per_action = 1       #4
        self.update_target_freq = 1000          #3000
        self.timestep_per_train = 100  # Number of timesteps between training interval       # 100

        # Initialize Atoms
        self.num_atoms = num_atoms  # 51 for C51
        self.v_max = 10*math.sqrt(2)  # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        self.v_min = -10*math.sqrt(2)  # -0.1*26 - 1 = -3.6
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z1 = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]
        self.z2 = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # Create replay memory using deque
        self.memory = deque()

        # at first i was like: let's make this number 10e7! But then I calculated how much memory 10e7 images require...
        self.max_memory = 25000  # number of previous transitions to remember

        # Models for value distribution
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.stats_window_size = 50  # window size for computing rolling statistics

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, image, data):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            # print("----------Random Action----------")
            action_1, action_2 = random.randrange(self.action_size), random.randrange(self.action_size)
        else:
            action_1, action_2 = self.get_optimal_action(image, data)

        return action_1, action_2

    def get_optimal_action(self, image, data):
        """Get optimal action for a state
        """
        pred = self.model.predict([image[np.newaxis, :], data[np.newaxis, :]])
        z1, z2 = pred[:len(pred) // 2], pred[len(pred) // 2:]
        z1_concat = np.vstack(z1)
        z2_concat = np.vstack(z2)
        q1 = np.sum(np.multiply(z1_concat, np.array(self.z1)), axis=1)
        q2 = np.sum(np.multiply(z2_concat, np.array(self.z2)), axis=1)

        # Pick action with the biggest Q value
        action_1 = np.argmax(q1)
        action_2 = np.argmax(q2)
        return action_1, action_2

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, image, data, action_1, action_2, r, next_image, next_data, done, global_step):
        self.memory.append((image, data, action_1, action_2, r, next_image, next_data, done, global_step))
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
        data_inputs = np.zeros(((num_samples, self.data_size)))
        next_images = np.zeros(((num_samples,) + self.image_size))
        next_datas = np.zeros(((num_samples, self.data_size)))
        m_prob_1 = [np.zeros((num_samples, self.num_atoms)) for i in range(action_size)]
        m_prob_2 = [np.zeros((num_samples, self.num_atoms)) for i in range(action_size)]
        action_1, action_2, reward, done = [], [], [], []

        for i in range(num_samples):
            image_inputs[i, :, :, :] = replay_samples[i][0]
            data_inputs[i, :] = replay_samples[i][1]
            action_1.append(replay_samples[i][2])
            action_2.append(replay_samples[i][3])
            reward.append(replay_samples[i][4])
            next_images[i, :, :, :] = replay_samples[i][5]
            next_datas[i, :] = replay_samples[i][6]
            done.append(replay_samples[i][7])

        prediction = self.model.predict([next_images, next_datas])
        prediction_ = self.target_model.predict([next_images, next_datas])
        z1, z2 = prediction[:len(prediction)//2], prediction[len(prediction)//2:]
        z1_, z2_ = prediction_[:len(prediction_)//2], prediction_[len(prediction_)//2:] # Return a list [32x51, 32x51, 32x51]

        # Get Optimal Actions for the next states (from distribution z)
        z1_concat = np.vstack(z1)
        z2_concat = np.vstack(z2)
        q1 = np.sum(np.multiply(z1_concat, np.array(self.z1)), axis=1)  # length (num_atoms x num_actions)
        q1 = q1.reshape((num_samples, action_size), order='F')
        q2 = np.sum(np.multiply(z2_concat, np.array(self.z2)), axis=1)  # length (num_atoms x num_actions)
        q2 = q2.reshape((num_samples, action_size), order='F')
        optimal_action_idxs_1 = np.argmax(q1, axis=1)
        optimal_action_idxs_2 = np.argmax(q2, axis=1)
        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):
            if done[i]:  # Terminal State
                # Distribution collapses to a single point
                Tz_1 = min(self.v_max, max(self.v_min, reward[i]))
                bj_1 = (Tz_1 - self.v_min) / self.delta_z
                m_l_1, m_u_1 = math.floor(bj_1), math.ceil(bj_1)
                m_prob_1[action_1[i]][i][int(m_l_1)] += (m_u_1 - bj_1)
                m_prob_1[action_1[i]][i][int(m_u_1)] += (bj_1 - m_l_1)

                Tz_2 = min(self.v_max, max(self.v_min, reward[i]))
                bj_2 = (Tz_2 - self.v_min) / self.delta_z
                m_l_2, m_u_2 = math.floor(bj_2), math.ceil(bj_2)
                m_prob_2[action_2[i]][i][int(m_l_2)] += (m_u_2 - bj_2)
                m_prob_2[action_2[i]][i][int(m_u_2)] += (bj_2 - m_l_2)
            else:
                for j in range(self.num_atoms):
                    Tz_1 = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z1[j]))
                    bj_1 = (Tz_1 - self.v_min) / self.delta_z
                    m_l_1, m_u_1 = math.floor(bj_1), math.ceil(bj_1)
                    m_prob_1[action_1[i]][i][int(m_l_1)] += z1_[optimal_action_idxs_1[i]][i][j] * (m_u_1 - bj_1)
                    m_prob_1[action_1[i]][i][int(m_u_1)] += z1_[optimal_action_idxs_1[i]][i][j] * (bj_1 - m_l_1)

                    Tz_2 = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z2[j]))
                    bj_2 = (Tz_2 - self.v_min) / self.delta_z
                    m_l_2, m_u_2 = math.floor(bj_2), math.ceil(bj_2)
                    m_prob_2[action_2[i]][i][int(m_l_2)] += z2_[optimal_action_idxs_2[i]][i][j] * (m_u_2 - bj_2)
                    m_prob_2[action_2[i]][i][int(m_u_2)] += z2_[optimal_action_idxs_2[i]][i][j] * (bj_2 - m_l_2)
        loss = self.model.fit([image_inputs, data_inputs], m_prob_1 + m_prob_2, batch_size=self.batch_size, epochs=2, verbose=0)
        return loss.history['loss']

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    weights_name = "c51_ddqn.h5"

    # Parameters
    action_size = 41
    MAX_EP = 1000000
    MAX_EP_STEP = 2500
    data_size = 2

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    img_rows, img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 3  # We stack 4 frames   # I use 1 RGB frame
    # C51
    num_atoms = 51
    image_size = (img_rows, img_cols, img_channels)
    agent = C51Agent(image_size, data_size, action_size, num_atoms)
    agent.model = Networks.value_distribution_network(image_size, data_size, num_atoms, action_size, agent.learning_rate)
    agent.target_model = Networks.value_distribution_network(image_size, data_size, num_atoms, action_size, agent.learning_rate)
    if weights_name in os.listdir(os.getcwd()):
        agent.load_model(weights_name)
    # Start training
    epsilon = agent.initial_epsilon
    # Buffer to compute rolling statistics
    scene_path = os.path.dirname(os.getcwd()) + '/scenes/diploma.ttt'
    env = Q_learning(headless=False, scene_path=scene_path)
    global_step = 0
    for episode in range(MAX_EP):
        env.reset()
        data, image = env.read_data(step=False)
        ep_r = 0
        for step in range(MAX_EP_STEP):
            global_step += 1
            a1, a2 = agent.get_action(image, data)
            next_data, next_image, r, done = env.step([a1, a2])  # make step in environment
            if not done:
                done = True if step == MAX_EP_STEP - 1 else False
            ep_r += r
            agent.replay_memory(image, data, a1, a2, r, next_image, next_data, done, global_step)

            # train
            if global_step > agent.observe and global_step % agent.timestep_per_train == 0:
                loss = agent.train_replay()

            image, data = next_image, next_data
            if global_step % 10000 == 0:
                print("Now we save model")
                agent.model.save_weights("c51_ddqn.h5", overwrite=True)
            if done:
                print("Episode: {0} Reward: {1}".format(episode, ep_r))
                break
