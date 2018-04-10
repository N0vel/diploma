import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import math
import numpy as np
import tensorflow as tf
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer
from keras.preprocessing.sequence import pad_sequences
from maze_env import Q_learning
import _pickle as pickle
from field_env import Environment
import random


def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    scene_path = os.path.dirname(os.getcwd()) + '/scenes/field.ttt'
    BUFFER_SIZE = 25000
    BATCH_SIZE = 1000
    UPDATE_FREQUENCY = 50
    gamma = 0.99  # gamma for target
    # if not working, try smaller learning rate
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.00001  # Learning rate for Actor
    LRC = 0.0001  # Learning rate for Critic
    action_dim = 2  #num of joints being controlled
    state_shape = (64, 64, 4)  #num of features in state

    EXPLORE = 1000.*50
    episode_count = pow(10, 10) if (train_indicator) else 1
    step = 0
    # epsilon = 1.0 if (train_indicator) else 0.0
    epsilon = 0.0
    max_steps = 2500

    #Tensorflow GPU optimization
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    from keras import backend as K
    K.set_learning_phase(train_indicator)
    K.set_session(sess)
    with tf.device('/gpu:0'):
        actor = ActorNetwork(sess, state_shape, action_dim, BATCH_SIZE, TAU, LRA)
        critic = CriticNetwork(sess, state_shape, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    episode_buff = list()
    env = Environment(headless=False, scene_path=scene_path)
    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        with open("buff.pickle", 'rb') as f:
            buff = pickle.load(f)
        print("Weight load successfully")
    except:
        print("Cannot find the weight")
    i = 0
    while True:
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        env.reset()
        if not train_indicator:
            print("start recording now")
            time.sleep(5)
        total_reward = 0.
        s_t = env.read_data()
        for j in range(max_steps):
            loss = 0
            epsilon -= 0.3 / EXPLORE
            if random.random() > epsilon:
                a_type = "Exploit"
                a_t = actor.model.predict(s_t).flatten()
            else:
                a_type = "Explore"
                a_t = np.random.uniform(0, 1, size=action_dim)
            s_t1, r_t, done = env.step(a_t)
            episode_buff.append([s_t, a_t, r_t, s_t1, done])
            s_t = s_t1
            # Do the batch update
            if j % UPDATE_FREQUENCY == 0 and buff.count() > 0:
                batch = buff.getBatch(BATCH_SIZE)
                states = np.asarray([e[0] for e in batch]).reshape((len(batch), 64, 64, 4))
                # actions = np.asarray([e[1].flatten() for e in batch]).reshape((len(batch), action_dim))
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch]).reshape((len(batch), 64, 64, 4))
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[2] for e in batch])
                predictions = actor.target_model.predict(new_states)
                target_q_values = critic.target_model.predict([new_states, predictions])
                for k in range(len(batch)):
                    if not dones[k]:
                        y_t[k] = rewards[k] + gamma * target_q_values[k]
                if (train_indicator):
                    loss += critic.model.train_on_batch([states, actions], y_t)
                    a_for_grad = actor.model.predict(states)
                    grads = critic.gradients(states, a_for_grad)
                    if np.sum(grads) == 0:
                        print('Zero gradients')
                    actor.train(states, grads)
                    actor.target_train()
                    critic.target_train()
                    if math.isnan(loss) or math.isinf(loss):
                        import sys
                        sys.exit()
                print("Episode", i, "Step", step, "Action", a_type, "Reward %.3f" % r_t, "Loss %.3f" % loss,
                      "Epsilon %.3f" % epsilon, "Time %.2f" % (j * 0.05), "Action", a_t)
            total_reward += r_t
            step += 1
            if done or j == max_steps - 1:
                if total_reward != 0:
                    value = 0
                    value_buf = []
                    for r in episode_buff[::-1]:
                        value = r[2] + gamma * value
                        value_buf.append(value)
                    value_buf.reverse()
                    for v in range(len(episode_buff)):
                        buff.add(episode_buff[v][0], episode_buff[v][1], value_buf[v], episode_buff[v][3],
                                 episode_buff[v][4])
                else:
                    value_buf = [-1.] * len(episode_buff)
                    for v in range(len(episode_buff)):
                        buff.add(episode_buff[v][0], episode_buff[v][1], value_buf[v], episode_buff[v][3],
                                 episode_buff[v][4])
                episode_buff[:] = []
                break

        if (train_indicator) and i % 50 == 0:
            print("Now we save model")
            actor.model.save_weights("actormodel.h5", overwrite=True)
            critic.model.save_weights("criticmodel.h5", overwrite=True)
            with open('buff.pickle', 'wb') as f:
                pickle.dump(buff, f)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        i += 1


    env.done()
    print("Finish.")

if __name__ == "__main__":
    playGame(1)
