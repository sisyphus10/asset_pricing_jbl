from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import keras
import keras as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers as L

from src.util import get_adam

Adam = get_adam()


def replace_nan_layer():
    """
    Creates a Keras Lambda layer for replacing specific values in the input.

    This function returns a Keras Lambda layer that is designed to replace all occurrences
    of a specific value (-99.99) in its input tensor with another value (0.0).

    The function uses TensorFlow's `where` function to identify elements in the input tensor
    that are equal to -99.99 and replace them with 0.0, while leaving other elements unchanged.

    Returns:
        keras.layers.Layer: A Keras Lambda layer that can be incorporated into a Keras model.
        When applied to an input tensor, it replaces every instance of -99.99 in the tensor
        with 0.0.
    """
    return L.Lambda(
        lambda x: tf.where(tf.equal(x, -99.99), 0.0, x),
    )


def build_simple_neural_net(
    output_units: int = 1,
    add_replace_nan_layer: bool = False,
    dropout_probability: float = 0.05,
    num_hidden_units: int = 64,
    compile: bool = True,
) -> keras.Model:
    layers = [replace_nan_layer()] if add_replace_nan_layer else []
    layers.extend(
        [
            L.AlphaDropout(dropout_probability),
            L.Dense(num_hidden_units, activation="relu"),
            L.AlphaDropout(dropout_probability),
            L.Dense(num_hidden_units, activation="relu"),
            L.AlphaDropout(dropout_probability),
            L.Dense(output_units),
        ],
    )
    model = keras.Sequential(layers)
    if compile:
        model.compile(optimizer=Adam(), loss="mse")
    return model


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


class ProgressTracker:
    """A class that estimates and tracks progress of a process."""

    _step_idx: int
    """At which step in a process we are currently at."""

    _step_times: list[float]
    """A list of times it took to complete each step."""

    _last_step_start_time: float
    """The time when the last step was completed."""

    def __init__(self, total_expected_number_of_steps: int):
        self.reset()
        self._total_steps = total_expected_number_of_steps

    def reset(self):
        self._step_idx = 0
        self._step_times = []
        self._last_step_start_time = None

    def _get_current_time(self):
        # return time.time()
        return datetime.now()

    def step(self):
        """Mark the completion of a step."""
        if self._last_step_start_time is not None:
            self._step_times.append(
                pd.Timestamp.now() - self._last_step_start_time,
            )

        self._last_step_start_time = pd.Timestamp.now()
        self._step_idx += 1


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = L.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [L.Dense(ff_dim, activation="relu"), L.Dense(embed_dim)],
        )
        self.layernorm1 = L.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = L.LayerNormalization(epsilon=1e-6)
        self.dropout1 = L.Dropout(rate)
        self.dropout2 = L.Dropout(rate)

    def call(self, x, training):
        inputs: tf.Tensor = x[0]
        attention_mask: tf.Tensor = x[1]

        attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class SimpleTransformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        dropout_rate,
        output_units,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = L.Dropout(dropout_rate)
        self.flatten = L.Flatten()
        self.final_layer = L.Dense(output_units)

    def call(self, inputs, training=None, **kwargs):
        x = inputs[0]
        attention_mask = inputs[1]

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](
                (x, attention_mask),
                training=training,
            )
        # x = self.flatten(x)
        return self.final_layer(x)


def build_transformer_model(
    num_assets: int = 8192,
    num_features: int = 46,
    num_layers: int = 2,
    d_model: int = 64,
    num_heads: int = 4,
    dff: int = 128,
    output_units: int = 1,
    add_replace_nan_layer: bool = False,
    dropout_probability: float = 0.1,
    compile: bool = False,
) -> tf.keras.Model:
    # Takes a tuple of (features, asset_mask) as input
    # Check, if we need to tell tf that they have the same first dim?!
    features = tf.keras.Input(shape=(num_assets, num_features))

    # shape: (n_time_steps, n_assets)
    asset_mask = tf.keras.Input(shape=(num_assets,))
    attention_mask = tf.keras.layers.Lambda(
        lambda x: tf.cast(x[..., :, None] * x[..., None, :], tf.float32),
    )(asset_mask)

    # np.logical_and(mask[:, None, :], mask[:, :, None]).astype(float)

    transformer = SimpleTransformer(
        num_layers,
        d_model,
        num_heads,
        dff,
        dropout_probability,
        output_units,
    )

    # Use x as the internal state
    x = features

    layers = [replace_nan_layer()] if add_replace_nan_layer else []
    layers.append(L.Dense(d_model, activation="linear"))

    for layer in layers:
        x = layer(x)

    # Mask the activations
    x = tf.multiply(
        x,
        asset_mask[..., :, None],
        name="pointwise_masking_inputs",
    )
    outputs = transformer((x, attention_mask), training=True)
    outputs = tf.multiply(
        outputs,
        asset_mask[..., :, None],
        name="pointwise_masking_outputs",
    )

    model = tf.keras.Model(inputs=(features, asset_mask), outputs=outputs)

    if compile:
        raise NotImplementedError("Should not be used in a compiled fashion")
        model.compile(optimizer=Adam(), loss="mse")

    return model


# And we need some functions for padding the inputs
# and also for removing the padding after the model ran
