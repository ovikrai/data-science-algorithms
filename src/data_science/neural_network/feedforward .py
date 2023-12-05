import numpy as np
import tensorflow as tf
from algos.model import Model


class FeedForward(Model):
    neural_network: tf.keras.models.Sequential
    UNITS = 0
    ACTIVATION = 1

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 dense_layers: list[tuple[int, str]],
                 optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy']
                 ):
        super().__init__(x_train, x_test, y_train, y_test)

        # EXAMPLE [ (10 , 'relu') ... (10, 'sigmoid') ]
        self.neural_network = tf.keras.models.Sequential()

        # ADDING N-LAYERS TO THE SEQUENTIAL MODEL
        n = len(dense_layers)
        for i in range(0, n):
            self.neural_network[i].add(tf.keras.layers.Dense(
                units=dense_layers[i][self.UNITS],
                activation=dense_layers[i][self.ACTIVATION]
            ))

            # FINAL DENSE LAYER
            self.neural_network.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # COMPILE NEURAL NETWORK
        self.neural_network.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train(self, batch_size=32, epochs=100):
        self.neural_network.fit(self.x_train, self.y_train,
                                batch_size=batch_size, epochs=epochs)

    def predict(self):
        return self.neural_network.predict(self.x_test)
