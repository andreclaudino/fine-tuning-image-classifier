import os.path
from typing import List, Callable, Tuple

import tensorflow as tf


def load_jpg_image_dataset(source_glob: str, batch_size: int, shuffle: bool, label_class_names: List[str],
                           path_segment_class_index: int, image_channels: int,
                           image_output_size: Tuple[int, int]) -> tf.data.Dataset:
    files_list = tf.data.Dataset.list_files(source_glob, shuffle=shuffle)

    label_extractor = _make_label_extractor(label_class_names, path_segment_class_index)
    image_loader = _make_jeg_image_loader(image_channels, image_output_size)

    processor = _make_path_to_dataset_processor(image_loader, label_extractor)

    dataset = files_list.map(processor).batch(batch_size)

    return dataset


def _make_path_to_dataset_processor(
        image_loader: Callable[[tf.Tensor], tf.Tensor],
        label_extractor: Callable[[tf.Tensor], tf.Tensor]) -> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:

    def images_and_label_loader(file_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = image_loader(file_path)
        label = label_extractor(file_path)
        return image, label

    return images_and_label_loader


def _make_jeg_image_loader(channels: int, image_size: Tuple[int, int]) -> Callable[[tf.Tensor], tf.Tensor]:

    def image_loader(source_path: tf.Tensor) -> tf.Tensor:
        file_data = tf.io.read_file(source_path)
        image = tf.io.decode_jpeg(file_data, channels=channels)
        resized_image = tf.image.resize(image, image_size)
        return resized_image

    return image_loader


def _make_label_extractor(label_class_names: List[str],
                          path_segment_class_index: int) -> Callable[[tf.Tensor], tf.Tensor]:

    def get_label(file_path: tf.Tensor) -> tf.Tensor:
        parts = tf.strings.split(file_path, os.path.sep)
        label_vector = parts[path_segment_class_index] == label_class_names
        class_index = tf.cast(label_vector, dtype=tf.int16)
        return class_index

    return get_label
