from __future__ import print_function

from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, Adadelta
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input, ConvLSTM2D, LSTM, Reshape, AveragePooling2D, Lambda, Merge, Activation, Embedding, concatenate
from keras import backend as K
class Networks(object):
    @staticmethod
    def value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
        """Model Value Distribution
                    With States as inputs and output Probability Distributions for all Actions
                    """
        image_input = Input(shape=(input_shape))
        cnn_feature = Conv2D(256, (7, 7), strides=4, padding='same', activation='elu')(image_input)
        cnn_feature = Conv2D(256, (5, 5), strides=2, padding='same', activation='elu')(cnn_feature)
        cnn_feature = Reshape((1, 8, 8, 256))(cnn_feature)
        cnn_feature = ConvLSTM2D(512, (3, 3), padding='same', stateful=False)(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)
        cnn_feature = Dense(512, activation='elu')(cnn_feature)
        distribution_list = []
        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation='softmax')(cnn_feature))
        model = Model(inputs=image_input, outputs=distribution_list)
        adam = Adadelta(lr=1.0)
        model.compile(loss='categorical_crossentropy', optimizer=adam)
        model.summary()
        return model
