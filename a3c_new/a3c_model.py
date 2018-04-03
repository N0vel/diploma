import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.models import Model


def build_policy_and_value_networks(num_actions, agent_history_length, resized_width, resized_height):
    state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])

    inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
    shared = Conv2D(name="conv1", activation="relu", kernel_size=(8, 8), filters=32, strides=(4, 4), padding="same")(
        inputs)
    shared = Conv2D(name="conv2", activation="relu", kernel_size=(4, 4), filters=64, strides=(2, 2), padding="same")(
        shared)
    shared = Flatten()(shared)
    shared = Dense(name="h1", activation="relu", units=256)(shared)
    shared = Dense(name="h2", activation="relu", units=256)(shared)

    action_probs_1 = Dense(name="p1", activation="softmax", units=num_actions)(shared)
    action_probs_2 = Dense(name="p2", activation="softmax", units=num_actions)(shared)

    state_value = Dense(name="v", activation="linear", units=1)(shared)

    policy_network = Model(inputs=inputs, outputs=[action_probs_1, action_probs_2])
    value_network = Model(inputs=inputs, outputs=state_value)

    p_params = policy_network.trainable_weights
    v_params = value_network.trainable_weights

    p_out = policy_network(state)
    v_out = value_network(state)
    return state, p_out, v_out, p_params, v_params