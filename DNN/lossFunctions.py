
import tensorflow as tf
from keras import backend as K


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))


def quantile_binary_crossentropy(y_true, y_pred):
    q = 75
    perc = tf.contrib.distributions.percentile(y_pred, q, interpolation='lower')
    perc += tf.contrib.distributions.percentile(y_pred, q, interpolation='higher')
    perc /= 2.

    y_pred_Pecetile = K.switch(y_pred >= perc, y_pred*1.0, y_pred*0.001)

    loss = K.mean(K.binary_crossentropy(y_true, y_pred_Pecetile, from_logits=True))
    return loss


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred-y_true))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

