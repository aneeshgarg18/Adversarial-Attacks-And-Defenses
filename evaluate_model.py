import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_model(model_fn, X, Y, epses):
    """
    Evaluate a model on given examples.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param X: testing examples (4-d tensor).
    :param Y: one-hot true labels (2-d tensor).
    :return: list of prediction accuracies at epses
    """
    l = []
    for eps in tqdm(epses):
        yp = tf.argmax(model_fn(x), axis=1)
        yt = tf.argmax(Y, axis=1)
        acc = np.sum(yp == yt) / np.sum(yt == yt)
        l += [a]
    return l