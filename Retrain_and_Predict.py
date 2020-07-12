import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import Analysis as An


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], 100*window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def retrain():
    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1

    data = An.weekly_analysis()
    ser = data['change'][1:].to_numpy()/100
    print()
    model = tf.keras.models.load_model(r"Models/M_20_10000_epochs_retrain")
    print(np.expand_dims(np.expand_dims(ser[-20:], axis=-1), axis=0).shape)

    series_len = (len(data['change'][1:]) // batch_size) * batch_size
    series = data['change'][1:][-series_len:].to_numpy() / 100
    train_data = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)

    class CallbackClass(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("mae") < 0.4:
                print("got less that 0.1")
                self.model.stop_training = True

    callbacks = CallbackClass()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mse', optimizer=optimizer, metrics='mae')
    history = model.fit(train_data, epochs=1000, callbacks=[callbacks])
    plt.semilogx(history.history["loss"])
    plt.show()
    model.save(r"Models/M_20_10000_epochs_retrain")

    def pred(x):
        return np.squeeze(model.predict(np.expand_dims(np.expand_dims(x, axis=-1), axis=0))).item()
    data['pred'] = (data['change']/100).rolling(window=window_size).apply(lambda x: pred(x))
    data.to_csv('heck_20_retrain.csv')


if __name__ == '__main__':
    retrain()
