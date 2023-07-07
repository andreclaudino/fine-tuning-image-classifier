import logging

import tensorflow as tf

from fine_tunner.models.image_classifier.model import ImageClassifier
from fine_tunner.persistence.metadata.metrics_exporter import MetricsExporter


def train_model(model: ImageClassifier, learning_rate: float, train_dataset: tf.data.Dataset,
                evaluation_dataset: tf.data.Dataset, train_epochs: int,
                training_metrics_exporter: MetricsExporter, report_step: int) -> ImageClassifier:

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(train_epochs):
        logging.info(f"Starging epoch {epoch}")

        for step, (batch_inputs, batch_labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(batch_inputs, training=True)
                loss_value = tf.keras.losses.categorical_crossentropy(batch_labels, logits)
            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            if step % report_step == 0:
                mean_loss = tf.reduce_mean(loss_value)
                logging.info(f"epoch = {epoch:4d} step = {step:5d} "
                             f"categorical cross entroypy = {mean_loss:8.8f}")

        evaluation_loss = tf.keras.metrics.CategoricalCrossentropy()
        for step, (batch_inputs, batch_labels) in enumerate(evaluation_dataset):
            logits = model(batch_inputs)
            evaluation_loss.update_state(batch_labels, logits)

        evaluation_loss_value = evaluation_loss.result()
        mean_evaluation_loss_value = tf.reduce_mean(evaluation_loss_value)
        logging.info(f"test epoch = {epoch:4d} step = {step:5d} "
                     f"categorical cross entroypy = {mean_evaluation_loss_value:8.8f}")
        evaluation_loss.reset_state()

    return model

