from __future__ import print_function

from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding, concatenate
from keras.optimizers import SGD, Adam, rmsprop, Adadelta

class Networks(object):

    @staticmethod
    def value_distribution_network(input_shape, data_size, num_atoms, action_size, learning_rate):
        """Model Value Distribution
        With States as inputs and output Probability Distributions for all Actions
        """

        image_input = Input(shape=(input_shape))
        cnn_feature = Conv2D(64, (7, 7), strides=4, padding='same', activation='elu')(image_input)
        cnn_feature = Conv2D(64, (5, 5), strides=2, padding='same', activation='elu')(cnn_feature)
        cnn_feature = Conv2D(64, (3, 3), strides=2, padding='same', activation='elu')(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)

        d_input = Input(shape=(data_size,))
        dense_1 = Dense(256, activation='tanh')(d_input)
        dense_2 = Dense(256, activation='tanh')(dense_1)
        concat = concatenate([cnn_feature, dense_2])
        cnn_feature = Dense(512, activation='elu')(concat)

        distribution_list = []
        for i in range(action_size*2):
            distribution_list.append(Dense(num_atoms, activation='softmax')(cnn_feature))
        model = Model(inputs=[image_input, d_input], outputs=distribution_list)
        adam = Adadelta(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

        return model