import numpy as np
import tensorflow as tf
from algos.model import Model


class Recurrent(Model):
    neural_network: tf.keras.models.Sequential

    # LSTM AND DROPOUT LAYER
    UNITS = 0
    RETURN_SEQUENCE = 1
    ACTIVATION = 2
    RECURRENT_ACTIVATION = 3
    DROP_RATE = 4

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 lstm_layers: list[tuple[int, bool, str, str, bool]],
                 dense_layers: list[tuple[int, str]],
                 optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'],
                 ):
        super().__init__(x_train, x_test, y_train, y_test)

        # EXAMPLE [ (10 , 'relu', ..., 0.2) ... ('pool', 10, 2, ... ,0.2) ]
        #  ADDING INPUT LAYER
        self.neural_network = tf.keras.models.Sequential()
        self.neural_network.add(tf.keras.layers.LSTM(
            units=lstm_layers[0][self.UNITS],
            return_sequences=lstm_layers[0][self.RETURN_SEQUENCE],
            activation=lstm_layers[0][self.ACTIVATION],
            recurrent_activation=lstm_layers[0][self.RECURRENT_ACTIVATION],
            input_shape=(self.x_train.shape[1], 1)
        ))
        self.neural_network.add(tf.keras.layers.Dropout(rate=lstm_layers[0][self.DROP_RATE]))

        # ADDING N-1 LSTM AND DROPOUT LAYERS
        n = len(lstm_layers)
        for i in range(1, n):
            self.neural_network.add(tf.keras.layers.LSTM(
                units=lstm_layers[i][self.UNITS],
                return_sequences=lstm_layers[i][self.RETURN_SEQUENCE],
                activation=lstm_layers[i][self.ACTIVATION],
                recurrent_activation=lstm_layers[i][self.RECURRENT_ACTIVATION],
            ))
            self.neural_network.add(tf.keras.layers.Dropout(rate=lstm_layers[i][self.DROP_RATE]))

        # FINAL OUTPUT DENSE LAYER
        self.neural_network.add(tf.keras.layers.Dense(units=1))

        # COMPILE NEURAL NETWORK
        self.neural_network.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train(self, epochs=100, batch_size=32):
        self.neural_network.fit(self.x_train,
                                self.y_train,
                                epochs=epochs,
                                batch_size=batch_size)

    def predict(self):
        return self.neural_network.predict(self.x_test)
