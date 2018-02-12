import time
import math
import numpy as np
import tensorflow as tf
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer, InputBuffer
from keras.preprocessing.sequence import pad_sequences
from q_learning import Q_learning
import _pickle as pickle

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 1000000
    BATCH_SIZE = 32
    GAMMA = 0.99
    # if not working, try smaller learning rate
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.00000001  # Learning rate for Actor
    LRC = 0.0000001  # Learning rate for Critic
    action_dim = 2  #num of joints being controlled
    state_dim = 9  #num of features in state

    EXPLORE = 500.0*50
    episode_count = pow(10, 10) if (train_indicator) else 1
    reward = 0
    done = False
    have_to_reset = False
    step = 0
    epsilon = 0.6 if (train_indicator) else 0.0
    max_steps = 5000

    #Tensorflow GPU optimization
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    from keras import backend as K
    K.set_learning_phase(train_indicator)
    K.set_session(sess)
    with tf.device('/gpu:0'):
        actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
        critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    episode_buff = list()
    # Generate a Torcs environment
    env = Q_learning(headless=False)

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
        for j in range(max_steps):
            loss = 0
            epsilon -= 0.3 / EXPLORE
            s_t = env.read_data(step=False)
            s_t = s_t.reshape((-1, 9))
            if np.random.random() > epsilon:
                a_type = "Exploit"
                a_t = actor.model.predict(s_t)
            else:
                a_type = "Explore"
                a_t = np.random.uniform(0, 1, size=action_dim)
            ob, r_t, done, have_to_reset = env.step(a_t)
            if j == 0:
                start_distance = env.distance
            s_t1 = ob.flatten()
            episode_buff.append([s_t, a_t, r_t, s_t1, done])
            # Do the batch update
            if j % 1 == 0 and buff.count() > 0:
                batch = buff.getBatch(BATCH_SIZE)
                states = np.asarray([e[0] for e in batch]).reshape((len(batch), state_dim))
                actions = np.asarray([e[1].flatten() for e in batch]).reshape((len(batch), action_dim))
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch]).reshape((len(batch), state_dim))
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[1].flatten() for e in batch]).reshape((len(batch), action_dim))
                predictions = actor.target_model.predict(new_states)
                target_q_values = critic.target_model.predict([new_states, predictions])
                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA * target_q_values[k]
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
            total_reward += r_t
            print("Episode", i, "Step", step, "Action", a_type, "Reward %.3f" % r_t, "Loss %.3f" % loss,
                  "Epsilon %.3f" %
                  epsilon, "Time %.2f" % (j * 0.05), "Distance %.3f" % env.distance, "Angle %.3f" % env.angle, "Action",
                  a_t)
            step += 1
            if have_to_reset or done or j == max_steps - 1:
                if done:
                    for i in range(len(episode_buff)):
                        """
                        Try to change +=
                        """
                        # episode_buff[i][2] += 100. * (start_distance - env.distance)/len(episode_buff)
                        buff.add(episode_buff[i][0], episode_buff[i][1], episode_buff[i][2], episode_buff[i][3],
                                 episode_buff[i][4])
                else:
                    for i in range(len(episode_buff)):
                        # episode_buff[i][2] += 100. * (start_distance - env.distance) / len(episode_buff) - 2.
                        episode_buff[i][2] -= 2.
                        buff.add(episode_buff[i][0], episode_buff[i][1], episode_buff[i][2], episode_buff[i][3],
                                 episode_buff[i][4])
                episode_buff = list()
                have_to_reset = False
                break
        if (train_indicator) and i % 10 == 0:
            print("Now we save model")
            actor.model.save_weights("actormodel.h5", overwrite=True)
            critic.model.save_weights("criticmodel.h5", overwrite=True)
            if i % 10 == 0:
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
