import tensorflow as tf
from fine_tunner.models.image_classifier.model import ImageClassifier


def save_trained_model(model: ImageClassifier, output_path: str) -> None:
    tf.saved_model.save(model, output_path)
