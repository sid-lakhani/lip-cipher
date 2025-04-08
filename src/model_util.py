from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, TimeDistributed, Bidirectional, LSTM, Dropout, Dense, Activation
import tensorflow as tf

def build_model():
    model = Sequential()
    model.add(Conv3D(128, 3, padding='same', input_shape=(75, 46, 140, 1)))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='Orthogonal')))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='Orthogonal')))
    model.add(Dropout(0.5))

    model.add(Dense(41, activation='softmax', kernel_initializer='he_normal'))
    return model

def save_model(model, path: str):
    model.save(path)

def load_saved_model(path: str) -> tf.keras.models.Model:
    return tf.keras.models.load_model(path)
