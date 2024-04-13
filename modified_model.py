from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

class ModifiedModel:
    def __init__(self, input_size, output_size):
        self.model = Sequential([
            Dense(100, activation='softsign', kernel_initializer='glorot_uniform', input_shape=(input_size,)),
            Dense(80, activation='softsign', kernel_initializer='glorot_uniform'),
            Dense(50, activation='softsign', kernel_initializer='glorot_uniform'),
            Dense(output_size, activation='softmax')
        ])

    def compile_model(self):
        self.model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, x_train, y_train, epochs, batch_size, validation_split, callbacks):
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks, verbose=2)
