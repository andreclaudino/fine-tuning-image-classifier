import os
from typing import Any, Union, Dict

import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter


class MetricsExporter:
    _summary_writer: SummaryWriter

    def __init__(self, ouput_folder: str):
        os.makedirs(ouput_folder, exist_ok=True)
        self._summary_writer = tf.summary.create_file_writer(ouput_folder)

    def write_scalar(self, metric_name: str, metric_value: Union[tf.Tensor, float, int], step: int) -> None:
        self._summary_writer.add_scalar(metric_name, metric_value, step, new_style=True)

    def write_scalars_dictionary(self, main_metric_name: str, metric_values: Dict[str, Union[tf.Tensor, float, int]],
                                 step: int) -> None:
        self._summary_writer.add_scalars(main_metric_name, metric_values, step)
