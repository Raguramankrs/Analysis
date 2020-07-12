import tensorflow as tf
import Analysis as An
import numpy as np
import matplotlib.pyplot as plt


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], 100*window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def weekly_sequence_model():
    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1
    data = An.weekly_analysis()
    series_len = (len(data['change'][1:])//batch_size)*batch_size
    series = data['change'][1:][-series_len:].to_numpy()/100
    train_data = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)
    print(train_data)
    week_model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                               input_shape=[None]),
        tf.keras.layers.Conv1D(filters=32, kernel_size=20, strides=1, padding='causal', activation='tanh',
                               input_shape=[None, None, 1]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh')),
        tf.keras.layers.Dense(100, activation='tanh'),
        tf.keras.layers.Dense(1, activation='tanh'),
        tf.keras.layers.Lambda(lambda x: x*10)
    ])
    """optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8*10**(epoch/20))
    week_model.compile(loss='mse', optimizer=optimizer,
                       metrics='mae')
    history = week_model.fit(train_data, epochs=100, callbacks=[lr_scheduler])
    plt.semilogx(history.history["lr"], history.history["loss"])
    # plt.axis(1e-8, 1e-2, 0, 60)
    plt.show()
    lr = float(input("Damn lr"))"""

    optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam()

    class CallbackClass(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("mae") < 0.1:
                print("got less that 0.1")
                self.model.stop_training = True

    callbacks = CallbackClass()

    week_model.compile(loss='mse', optimizer=optimizer,
                       metrics='mae')
    history = week_model.fit(train_data, epochs=10000, callbacks=[callbacks])
    plt.semilogx(history.history["loss"])
    plt.show()
    week_model.save(r"Models\M_32_100_10000_epochs")
    print(week_model.predict(np.expand_dims(np.expand_dims(series[-20:], axis=-1), axis=0)))
    print(series[-1])


if __name__ == '__main__':
    weekly_sequence_model()
