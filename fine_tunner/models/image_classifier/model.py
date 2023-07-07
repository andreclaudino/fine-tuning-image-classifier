from typing import List, Any, Callable, Union

import tensorflow as tf


class ImageClassifier(tf.keras.Model):
    _input_layers: tf.keras.layers.Layer
    _output_layer: tf.keras.layers.Layer

    def __init__(self, pre_trained_layer: tf.keras.layers.Layer, output_class_number: int,
                 activation: Union[Callable[[Any, int], Any], str],
                 non_trainable_layers_count: int = 0, *args, **kwargs):
        """
        :param pre_trained_layer: The pre-trained layer loaded from another model, will be used as the model input layer
        :param output_class_number: The number of output classes
        :param activation: The reference or string name to the activation function to be used in the model output layer
        :param non_trainable_layers_count: The number of non-trainable layers on the model beginning of the model
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self._input_layers = pre_trained_layer
        self._update_non_trainable(non_trainable_layers_count)
        self._output_layer = tf.keras.layers.Dense(units=output_class_number, activation=activation)

    def call(self, inputs: Any, training: Any = None, mask: Any = None) -> tf.Tensor:
        input_layers_prediction = self._input_layers(inputs)
        final_prediction = self._output_layer(input_layers_prediction)
        return final_prediction

    def _update_non_trainable(self, non_trainable_layers_count: int):
        last_trainable_layer = non_trainable_layers_count + 1

        for layer in self.layers[:-last_trainable_layer]:
            layer.trainable = False
