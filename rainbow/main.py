import numpy as np
import random
import tensorflow as tf
import os
# from batchEnv import BatchEnvironment
from replayMemory import ReplayMemory, PriorityExperienceReplay
from model import create_deep_q_network, create_duel_q_network, create_model, create_distributional_model
from agent import DQNAgent
from q_learning import Q_learning


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

NUM_FRAME_PER_ACTION = 4
UPDATE_FREQUENCY = 4 # do one batch update when UPDATE_FREQUENCY number of new samples come
TARGET_UPDATE_FREQENCY = 10000
REPLAYMEMORY_SIZE = 500000
MAX_EPISODE_LENGTH = 100000
RMSP_EPSILON = 0.01
RMSP_DECAY = 0.95
RMSP_MOMENTUM =0.95
MAX_EPISODE_LENGTH = 100000
NUM_FIXED_SAMPLES = 10000
NUM_BURN_IN = 50000
LINEAR_DECAY_LENGTH = 4000000
NUM_EVALUATE_EPISODE = 20
POSSIBLE_ACTIONS = np.linspace(-100, 100, 21)

def get_fixed_samples(env, num_samples):
    fixed_samples = []
    env.reset()
    for _ in range(0, num_samples):
        action = np.array([np.random.choice(POSSIBLE_ACTIONS), np.random.choice(POSSIBLE_ACTIONS)])
        new_state, reward, is_terminal = env.step(action)
        for state in new_state:
            fixed_samples.append(state)
    return np.array(fixed_samples)

'''
self.env.reset()
s = self.env.read_data(step=False)
s_, r, done = self.env.step(a)
'''
def main():
    seed = 10703
    input_shape = (9,)
    gamma = 0.99
    epsilon = 0.1
    learning_rate = 0.00025
    batch_size = 32
    num_iteration = 20000000
    eval_every = 0.001
    is_duel = 1             # Whether use duel DQN
    is_double = 1           # Whether use double DQN
    is_per = 1              # Whether use PriorityExperienceReplay
    is_distributional = 1   # Whether use distributional DQN
    num_step = 1            # Num Step for multi-step DQN, 3 is recommended
    is_noisy = 1            # Whether use NoisyNet


    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    env = Q_learning(headless=False)

    if is_per == 1:
        replay_memory = PriorityExperienceReplay(REPLAYMEMORY_SIZE, input_shape)
    else:
        replay_memory = ReplayMemory(REPLAYMEMORY_SIZE,input_shape)


    create_network_fn = create_deep_q_network if is_duel == 0 else create_duel_q_network
    create_model_fn = create_model if is_distributional == 0 else create_distributional_model
    noisy = True if is_noisy == 1 else False
    online_model, online_params = create_model_fn(input_shape, num_actions,
                    'online_model', create_network_fn, trainable=True, noisy=noisy)
    target_model, target_params = create_model_fn(input_shape, num_actions,
                    'target_model', create_network_fn, trainable=False, noisy=noisy)
    update_target_params_ops = [t.assign(s) for s, t in zip(online_params, target_params)]


    agent = DQNAgent(online_model,
                    target_model,
                    replay_memory,
                    num_actions,
                    gamma,
                    UPDATE_FREQUENCY,
                    TARGET_UPDATE_FREQENCY,
                    update_target_params_ops,
                    batch_size,
                    is_double,
                    is_per,
                    is_distributional,
                    num_step,
                    is_noisy,
                    learning_rate,
                    RMSP_DECAY,
                    RMSP_MOMENTUM,
                    RMSP_EPSILON)

    sess = tf.Session(config=tf.ConfigProto())
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        # make target_model equal to online_model
        sess.run(update_target_params_ops)

        print('Prepare fixed samples for mean max Q.')
        fixed_samples = get_fixed_samples(env, NUM_FIXED_SAMPLES)

        print('Burn in replay_memory.')
        agent.fit(sess, batch_environment, NUM_BURN_IN, do_train=False)
        
        # Begin to train:
        fit_iteration = int(num_iteration * eval_every)
        for i in range(0, num_iteration, fit_iteration):
            # Evaluate:
            reward_mean, reward_var = agent.evaluate(sess, batch_environment, NUM_EVALUATE_EPISODE)
            mean_max_Q = agent.get_mean_max_Q(sess, fixed_samples)
            print("%d, %f, %f, %f"%(i, mean_max_Q, reward_mean, reward_var))
            # Train:
            agent.fit(sess, batch_environment, fit_iteration, do_train=True)

    batch_environment.close()

if __name__ == '__main__':
    main()
