import logging
import os.path

import click

from fine_tunner.constants import PATH_SEGMENT_CLASS_INDEX, IMAGE_CHANNELS, IMAGE_SIZE, SHUFFLE_TRAIN_DATASET, \
    SHUFFLE_VALIDATION_DATASET, REPORT_STEP
from fine_tunner.models.image_classifier.model import ImageClassifier
from fine_tunner.parameter_utils import extract_parameters_from_comma_separated_string
from fine_tunner.persistence.data.loaders import load_jpg_image_dataset
from fine_tunner.persistence.metadata.metrics_exporter import MetricsExporter
from fine_tunner.persistence.models.model_savers import save_trained_model
from fine_tunner.persistence.models.pre_trained_loaders import load_default_pre_trained_mobile_net
from fine_tunner.training.model_trainer import train_model


@click.command()
@click.option("--activation-function", type=click.STRING, default="softmax")
@click.option("--learning-rate", type=click.FLOAT, default="0.0001")
@click.option("--mobilenet-ignored-layers", type=click.INT, default=1)
@click.option("--non-trainable-layers-count", type=click.INT, default=22)
@click.option("--training-dataset-path", type=click.STRING)
@click.option("--evaluation-dataset-path", type=click.STRING)
@click.option("--label-class-names", type=click.STRING)
@click.option("--train-epochs", type=click.INT, default=100)
@click.option("--output-path", type=click.STRING)
@click.option("--batch-size", type=click.INT, default=32)
def main(activation_function: str, learning_rate: float, mobilenet_ignored_layers: int,
         non_trainable_layers_count: int, training_dataset_path: str, evaluation_dataset_path: str,
         label_class_names: str, train_epochs: int, output_path: str, batch_size: int) -> None:

    logging.getLogger().setLevel(logging.INFO)

    label_class_names_list = extract_parameters_from_comma_separated_string(label_class_names)
    output_class_number = len(label_class_names_list)

    pre_trained_mobilenet = load_default_pre_trained_mobile_net(ignored_layers=mobilenet_ignored_layers)
    model = ImageClassifier(
        pre_trained_mobilenet, output_class_number=output_class_number, activation=activation_function,
        non_trainable_layers_count=non_trainable_layers_count,
    )

    metrics_exporter_dir = os.path.join(output_path, "metrics")
    training_metrics_exporter = MetricsExporter(metrics_exporter_dir)

    training_dataset = load_jpg_image_dataset(
        training_dataset_path, batch_size, SHUFFLE_TRAIN_DATASET, label_class_names_list,
        PATH_SEGMENT_CLASS_INDEX, IMAGE_CHANNELS, IMAGE_SIZE)

    evaluation_dataset = load_jpg_image_dataset(
        evaluation_dataset_path, batch_size, SHUFFLE_VALIDATION_DATASET, label_class_names_list,
        PATH_SEGMENT_CLASS_INDEX, IMAGE_CHANNELS, IMAGE_SIZE)

    trained_model = train_model(model, learning_rate, training_dataset, evaluation_dataset, train_epochs,
                                training_metrics_exporter, REPORT_STEP)

    model_output_path = os.path.join(output_path, "models", "model.h5")
    save_trained_model(trained_model, model_output_path)


if __name__ == '__main__':
    main()
