import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import Analysis as An


def predict():
    window_size = 20
    data = An.weekly_analysis()
    ser = data['change'][1:].to_numpy()/100
    print()
    model = tf.keras.models.load_model(r"Models/M_20_10000_epochs")
    print(np.expand_dims(np.expand_dims(ser[-20:], axis=-1), axis=0).shape)

    def pred(x):
        return np.squeeze(model.predict(np.expand_dims(np.expand_dims(x, axis=-1), axis=0))).item()
    data['pred'] = (data['change']/100).rolling(window=window_size).apply(lambda x: pred(x))
    data.to_csv('heck_20.csv')


if __name__ == '__main__':
    predict()
