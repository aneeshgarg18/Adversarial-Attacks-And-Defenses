import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_adv(model_fn, X, Y, eps=1, batch_size=1000):
    """
    Generate adversarial examples from given examples.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param X: testing examples (4-d tensor).
    :param Y: one-hot true labels (2-d tensor).
    :param Y: perturbation parameter.
    :return: adversarial examples
    """
    adv_x = X.copy()
    for i in range(X.shape[0]//batch_size):
        i1, i2 = i*batch_size, (i+1)*batch_size
        adv_x[i1:i2] = model_fn(model_fn, X[i1:i2], eps=eps, eps_iter=eps/10, nb_iter=10, y=Y[i1:i2])
    return adv_x