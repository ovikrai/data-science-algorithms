import numpy as np
import tensorflow as tf
from algos.model import Model


class Convolutional(Model):
    neural_network: tf.keras.models.Sequential

    # CONVOLUTIONAL LAYER RECORD INDEX POSITIONS
    CONV_FILTERS = 0
    CONV_KERNEL_SIZE = 1
    CONV_ACTIVATION = 2

    # POOL RECORD INDEX POSITIONS
    POOL_SIZE = 3
    POOL_STRIDES = 4

    # DENSE RECORD INDEX POSITIONS
    DENSE_UNITS = 0
    DENSE_ACTIVATION = 1

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 conv_layers: list[tuple[int, int, str, int, int]],
                 dense_layers: list[tuple[int, str]],
                 optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy']
                 ):
        super().__init__(x_train, x_test, y_train, y_test)

        # EXAMPLE [ (10 , 'relu', ) ... ('pool', 10, 2) ]
        #  ADDING INPUT LAYER
        self.neural_network = tf.keras.models.Sequential()
        self.neural_network.add(tf.keras.layers.Conv2D(
            filters=conv_layers[0][self.CONV_FILTERS],
            kernel_size=conv_layers[0][self.CONV_KERNEL_SIZE],
            activation=conv_layers[0][self.CONV_ACTIVATION],
            input_shape=[64, 64, 3]
        ))
        self.neural_network.add(tf.keras.layers.MaxPool2D(
            filters=conv_layers[0][self.POOL_SIZE],
            kernel_size=conv_layers[0][self.POOL_STRIDES]
        ))

        # ADDING N-1 convolutional and pooling layers
        n = len(conv_layers)
        for i in range(1, n):
            self.neural_network.add(tf.keras.layers.Conv2D(
                filters=conv_layers[i][self.CONV_FILTERS],
                kernel_size=conv_layers[i][self.CONV_KERNEL_SIZE],
                activation=conv_layers[i][self.CONV_ACTIVATION],
            ))
            self.neural_network.add(tf.keras.layers.MaxPool2D(
                filters=conv_layers[i][self.POOL_SIZE],
                kernel_size=conv_layers[i][self.POOL_STRIDES]
            ))

        # ADDING Flatten and M-Dense Layers
        self.neural_network.add(tf.keras.layers.Flatten())

        m = len(dense_layers)
        for i in range(0, m):
            self.neural_network[i].add(tf.keras.layers.Dense(
                units=dense_layers[i][self.DENSE_UNITS],
                activation=dense_layers[i][self.DENSE_ACTIVATION]
            ))

        # FINAL DENSE LAYER
        self.neural_network.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # COMPILE NEURAL NETWORK
        self.neural_network.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train(self, epochs=25):
        self.neural_network.fit(self.x_train,
                                validation_data=self.y_train,
                                epochs=epochs)

    def predict(self):
        return self.neural_network.predict(self.x_test)
