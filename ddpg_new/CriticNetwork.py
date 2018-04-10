import numpy as np
import math
from keras.layers import Dense, Flatten, Input, concatenate, Lambda, Conv2D, ConvLSTM2D, Activation, Reshape, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import Adadelta
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_shape,action_dim):
        print("Now we build the model")
        A = Input(shape=[action_dim], name='action2')
        S = Input(shape=state_shape)
        cnn_feature = Conv2D(128, (7, 7), strides=4, padding='same', activation='elu')(S)
        cnn_feature = Conv2D(128, (5, 5), strides=2, padding='same', activation='elu')(cnn_feature)
        cnn_feature = Reshape((1, 8, 8, 128))(cnn_feature)
        cnn_feature = ConvLSTM2D(256, (3, 3), padding='same', stateful=False)(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)
        cnn_feature = Dense(512, kernel_initializer='glorot_normal', activation='tanh')(cnn_feature)
        h = concatenate([cnn_feature, A])
        h2 = Dense(512, kernel_initializer='glorot_normal', activation='tanh')(h)
        h3 = Dense(512, kernel_initializer='glorot_normal', activation='tanh')(h2)
        V = Dense(1, activation="linear", kernel_initializer="glorot_normal")(h3)
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(optimizer=adam, loss='mse')
        return model, A, S 
