import tensorflow as tf


def dense(inputs,
           units,
           activation=tf.nn.relu,
           is_train=True,
           name=None):
    """ Fully-connected layer. """

    return tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=activation,
        trainable=is_train,
        name=name)


def dropout(inputs,
             keep_prob,
             name = None):
    """ Dropout layer. """
    return tf.nn.dropout(inputs,
                         keep_prob=keep_prob,
                         name=name)


def batch_norm(inputs,
                is_train=True,
                name = None):
    """ Batch normalization layer. """
    return tf.layers.batch_normalization(
        inputs=inputs,
        trainable=is_train,
        name=name)


def dense_bn_activation(inputs,
                        units,
                        activation=None,
                        is_train=True,
                        name=None):
    dense_layer = dense(inputs, units, activation, is_train, name)
    bn_layer = batch_norm(dense_layer, is_train, name)
    activation_layer = activation(bn_layer)
    return activation_layer
