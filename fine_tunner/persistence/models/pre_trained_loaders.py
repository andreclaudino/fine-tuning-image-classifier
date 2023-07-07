import tensorflow as tf


def load_default_pre_trained_mobile_net(ignored_layers: int) -> tf.keras.layers.Layer:
    """
    Load mobile net, ignoring the last layers indicated by ignoed_layers parameter
    :param ignored_layers: THe number of layers to be ignored at the end of the model
    :return:
    """
    mobile_net = tf.keras.applications.mobilenet_v2.MobileNetV2()
    last_layer_index = ignored_layers + 1
    last_layer = mobile_net.layers[-last_layer_index]
    return last_layer

