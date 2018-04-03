import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import threading
import tensorflow as tf
import numpy as np
import time
from a3c_model import build_policy_and_value_networks
from keras import backend as K
from field_env import Environment
import random

# Path params
EXPERIMENT_NAME = "diploma_a3c"
SUMMARY_SAVE_PATH = "summaries/"+EXPERIMENT_NAME
CHECKPOINT_SAVE_PATH = os.path.join(os.getcwd(), "./a3c.ckpt")
CHECKPOINT_NAME = os.path.join(os.getcwd(), "./a3c.ckpt")
CHECKPOINT_INTERVAL = 50
SUMMARY_INTERVAL = 25
TRAINING = True


# Experiment params
scene_path = os.path.dirname(os.getcwd()) + '/scenes/field.ttt'
ACTIONS = 41
POSSIBLE_ACTIONS = np.linspace(-1, 1, 41)
NUM_CONCURRENT = 4
EPSILON = 1.0
EPSILON_STEP = 0.00001


AGENT_HISTORY_LENGTH = 4
RESIZED_WIDTH = 64
RESIZED_HEIGHT = 64

# DQN Params
GAMMA = 0.99

# Optimization Params
# LEARNING_RATE = 0.00001
LEARNING_RATE = 0.001

# Shared global parameters
t_max = 2500
T = 0
TMAX = 1000000

def sample_policy_action(probs_1, probs_2):
    """
    Sample an action from an action probability distribution output by
    the policy network.
    """
    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial IT DOESN'T HELP
    probs_1 = probs_1.flatten()
    probs_2 = probs_2.flatten()

    # probs_1 = (probs_1 - np.finfo(np.float32).epsneg).flatten()
    # probs_2 = (probs_2 - np.finfo(np.float32).epsneg).flatten()
    # histogram_1 = np.random.multinomial(1, probs_1)
    # action_index_1 = int(np.nonzero(histogram_1)[0])
    # histogram_2 = np.random.multinomial(1, probs_2)
    # action_index_2 = int(np.nonzero(histogram_2)[0])

    action_index_1 = np.random.choice(np.array(range(ACTIONS)), p=probs_1)
    action_index_2 = np.random.choice(np.array(range(ACTIONS)), p=probs_2)

    return action_index_1, action_index_2


def actor_learner_thread(num, env, session, graph_ops, summary_ops, saver):
    # We use global shared counter T, and TMAX constant
    global TMAX, T, EPSILON, EPSILON_STEP

    # Unpack graph ops
    s, a1, a2, R, minimize, p_network, v_network = graph_ops

    # Unpack tensorboard summary stuff
    r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op = summary_ops

    # Wrap env with AtariEnvironment helper class
    time.sleep(5 * num)

    # Set up per-episode counters
    ep_reward = 0
    ep_avg_v = 0
    v_steps = 0

    while T < TMAX:
        env.reset()
        s_batch = []
        past_rewards = []
        a1_batch = []
        a2_batch = []
        t = 0
        t_start = t
        s_t = env.read_data()
        terminal = False
        while not (terminal or (t - t_start >= t_max)):
            # Perform action a_t according to policy pi(a_t | s_t)
            if random.random() > EPSILON:
                probs_1, probs_2 = session.run(p_network, feed_dict={s: [s_t]})
                action_index_1, action_index_2 = sample_policy_action(probs_1, probs_2)
            else:
                action_index_1, action_index_2 = random.choice(range(35, 41)), random.choice(range(35, 41))
            EPSILON -= EPSILON_STEP
            a_t1 = np.zeros([ACTIONS])
            a_t2 = np.zeros([ACTIONS])
            a_t1[action_index_1] = 1
            a_t2[action_index_2] = 1
            if t % 100 == 0:
                print("A1: {0} A2:{1} V: {2}".format(POSSIBLE_ACTIONS[action_index_1],
                                                     POSSIBLE_ACTIONS[action_index_2],
                                                     session.run(v_network, feed_dict={s: [s_t]})[0][0]))

            s_batch.append(s_t)
            a1_batch.append(a_t1)
            a2_batch.append(a_t2)

            s_t1, r_t, terminal = env.step(np.array([POSSIBLE_ACTIONS[action_index_1],
                                                  POSSIBLE_ACTIONS[action_index_2]]))
            ep_reward += r_t

            r_t = np.clip(r_t, -1, 1)
            past_rewards.append(r_t)

            t += 1
            s_t = s_t1

        if terminal:
            R_t = 0
        else:
            R_t = session.run(v_network, feed_dict={s: [s_t]})[0][0]  # Bootstrap from last state

        R_batch = np.zeros(t)
        for i in reversed(range(t_start, t)):
            R_t = past_rewards[i] + GAMMA * R_t
            R_batch[i] = R_t

        session.run(minimize, feed_dict={R: R_batch,
                                         a1: a1_batch,
                                         a2: a2_batch,
                                         s: s_batch})

        # Save progress
        if T % CHECKPOINT_INTERVAL == 0:
            saver.save(session, CHECKPOINT_SAVE_PATH, global_step=T)

        # Episode ended, collect stats and reset game
        session.run(update_ep_reward, feed_dict={r_summary_placeholder: ep_reward})
        print("THREAD:", num, "/ TIME", T, "/ REWARD", ep_reward, "/ EPSILON", EPSILON)
        # Reset per-episode counters
        ep_reward = 0
        T += 1


def build_graph():
    # Create shared global policy and value networks
    s, p_network, v_network, p_params, v_params = build_policy_and_value_networks(num_actions=ACTIONS,
                                                                                  agent_history_length=AGENT_HISTORY_LENGTH,
                                                                                  resized_width=RESIZED_WIDTH,
                                                                                  resized_height=RESIZED_HEIGHT)

    # Shared global optimizer
    optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE)

    # Op for applying remote gradients
    R_t = tf.placeholder("float", [None])
    a_t1 = tf.placeholder("float", [None, ACTIONS])
    a_t2 = tf.placeholder("float", [None, ACTIONS])
    log_prob_1 = tf.log(tf.reduce_sum(p_network[0] * a_t1, reduction_indices=1))
    log_prob_2 = tf.log(tf.reduce_sum(p_network[1] * a_t2, reduction_indices=1))
    p_loss_1 = -log_prob_1 * (R_t - v_network)
    p_loss_2 = -log_prob_2 * (R_t - v_network)
    p_loss = p_loss_1 + p_loss_2
    v_loss = tf.reduce_mean(tf.square(R_t - v_network))

    total_loss = p_loss + v_loss

    minimize = optimizer.minimize(total_loss)
    return s, a_t1, a_t2, R_t, minimize, p_network, v_network


# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Episode Reward", episode_reward)
    r_summary_placeholder = tf.placeholder("float")
    update_ep_reward = episode_reward.assign(r_summary_placeholder)
    ep_avg_v = tf.Variable(0.)
    tf.summary.scalar("Episode Value", ep_avg_v)
    val_summary_placeholder = tf.placeholder("float")
    update_ep_val = ep_avg_v.assign(val_summary_placeholder)
    summary_op = tf.summary.merge_all()
    return r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op


def train(session, graph_ops, saver):
    # Set up game environments (one per thread)
    envs = [Environment(headless=False, scene_path=scene_path) for i in range(NUM_CONCURRENT)]

    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]


    # Initialize variables
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_SAVE_PATH, session.graph)
    # Start NUM_CONCURRENT training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread,
                                              args=(thread_id, envs[thread_id], session, graph_ops, summary_ops, saver))
                             for thread_id in range(NUM_CONCURRENT)]
    for t in actor_learner_threads:
        t.start()

    last_summary_time = 0
    while True:
        now = time.time()
        if now - last_summary_time > SUMMARY_INTERVAL:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now
    for t in actor_learner_threads:
        t.join()


def evaluation(session, graph_ops, saver):
    saver.restore(session, CHECKPOINT_NAME)
    print("Restored model weights from ", CHECKPOINT_NAME)
    # Unpack graph ops
    s, a_t1, a_t2, R_t, minimize, p_network, v_network = graph_ops

    # Wrap env with AtariEnvironment helper class
    env = Environment(headless=False, scene_path=scene_path)

    for i_episode in range(100):
        env.reset()
        s_t = env.read_data()
        ep_reward = 0
        terminal = False
        while not terminal:
            # Forward the deep q network, get Q(s,a) values
            probs_1, probs_2 = p_network.eval(session=session, feed_dict={s: [s_t]})
            action_index_1, action_index_2 = sample_policy_action(probs_1, probs_2)
            s_t1, r_t, terminal, info = env.step(np.array([POSSIBLE_ACTIONS[action_index_1],
                                                           POSSIBLE_ACTIONS[action_index_2]]))
            s_t = s_t1
            ep_reward += r_t
        print(ep_reward)


def main(_):
    g = tf.Graph()
    with g.as_default(), tf.Session() as session:
        K.set_session(session)
        graph_ops = build_graph()
        saver = tf.train.Saver()
        if TRAINING:
            train(session, graph_ops, saver)
        else:
            evaluation(session, graph_ops, saver)


if __name__ == "__main__":
    tf.app.run()