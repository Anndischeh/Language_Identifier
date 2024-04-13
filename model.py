from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class BaseModel:
    def __init__(self, input_size, output_size):
        self.model = Sequential([
            Dense(100, activation='relu', kernel_initializer='he_normal', input_shape=(input_size,)),
            Dense(80, activation='relu', kernel_initializer='he_normal'),
            Dense(50, activation='relu', kernel_initializer='he_normal'),
            Dense(output_size, activation='softmax')
        ])

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, x_train, y_train, epochs, batch_size, validation_split):
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=2)
