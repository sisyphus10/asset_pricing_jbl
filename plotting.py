# %%
# TODO: @Bent: Why we increase dense layer of GAN from 46 to 64?
# %%
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from estimation import build_datasets
from src.data_loading import load_equity_data
from src.models import util
from src.models.gan_stuff import AdversarialSDF

# %%
# load data
sets, train = build_datasets()
df_train = train.data


# %%
class FirmChar:
    def __init__(self):
        self._category = [
            "Past Returns",
            "Investment",
            "Profitability",
            "Intangibles",
            "Value",
            "Trading Frictions",
        ]
        self.category2variables = {
            "Past Returns": [
                "r2_1",
                "r12_2",
                "r12_7",
                "r36_13",
                "ST_REV",
                "LT_Rev",
            ],
            "Investment": ["Investment", "NOA", "DPI2A", "NI"],
            "Profitability": [
                "PROF",
                "ATO",
                "CTO",
                "FC2Y",
                "OP",
                "PM",
                "RNA",
                "ROA",
                "ROE",
                "SGA2S",
                "D2A",
            ],
            "Intangibles": ["AC", "OA", "OL", "PCM"],
            "Value": [
                "A2ME",
                "BEME",
                "C",
                "CF",
                "CF2P",
                "D2P",
                "E2P",
                "Q",
                "S2P",
                "Lev",
            ],
            "Trading Frictions": [
                "AT",
                "Beta",
                "IdioVol",
                "LME",
                "LTurnover",
                "MktBeta",
                "Rel2High",
                "Resid_Var",
                "Spread",
                "SUV",
                "Variance",
            ],
        }
        self.variable2category = {}
        for category in self._category:
            for var in self.category2variables[category]:
                self.variable2category[var] = category
        self.category2color = {
            "Past Returns": "red",
            "Investment": "green",
            "Profitability": "grey",
            "Intangibles": "magenta",
            "Value": "purple",
            "Trading Frictions": "orange",
        }
        self.color2category = {
            value: key for key, value in self.category2color.items()
        }

    def getColorLabelMap(self):
        return {
            var: self.category2color[self.variable2category[var]]
            for var in self.variable2category
        }


class DataAttributes:
    def __init__(self):
        self.sets, self.data = build_datasets()
        self.firm_char = FirmChar()

    def get_firm_char(self):
        return self.firm_char

    def get_data(self):
        return self.data

    def get_feature_names(self):
        return self.data.feature_names

    def get_variable_categories(self):
        return self.firm_char.variable2category

    def get_variable_category(self, var):
        return self.firm_char.variable2category[var]

    def get_variable_color(self, var):
        return self.firm_char.category2color[
            self.firm_char.variable2category[var]
        ]

    def get_variable_color_category(self, color):
        return self.firm_char.color2category[color]

    def get_variable_category_color(self, category):
        return self.firm_char.category2color[category]

    def get_variable_category_variables(self, category):
        return self.firm_char.category2variables[category]

    def get_date_count_list(self):
        return len(self.data.mask)


# %%
data = DataAttributes()
frim_char = data.get_date_count_list()


# %%
def load_model(checkpoint_path, custom_objects=None):
    return tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={"SDF_Model": AdversarialSDF},
    )


checkpoint_path = (
    Path("/Users/johannes/PycharmProjects/asset-pricing")
    / "model_checkpoints"
    / "2024-06-20"
    / "run_1"
    / "AdversarialSDF_epoch_00.h5"
)

model = load_model(checkpoint_path, "AdversarialSDF")


print(model.summary())


# %%
# Feature importance plot based on the first dense layer weights
def plot_feature_importance(model, data):
    # Get the first dense layer and its weights
    first_dense_layer = model.layers[2]
    weights = first_dense_layer.get_weights()[0]

    # Calculate feature importance
    abs_weights = np.abs(weights)
    feature_importance = abs_weights.sum(axis=1)
    normalized_importance = feature_importance / feature_importance.sum()

    # Get feature names and create importance dictionary
    feature_names = data.get_feature_names()
    importance_dict = dict(zip(feature_names, normalized_importance))

    # Sort importance
    sorted_importance = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Get the color map
    color_map = data.get_firm_char().getColorLabelMap()

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.bar(
        range(len(sorted_importance)),
        [imp for _, imp in sorted_importance],
        color=[color_map[feat] for feat, _ in sorted_importance],
    )
    plt.xticks(
        range(len(sorted_importance)),
        [feat for feat, _ in sorted_importance],
        rotation=90,
    )
    plt.xlabel("Stock Attributes")
    plt.ylabel("Importance")
    plt.title("Stock Attribute Importance Based on First Layer Weights")
    plt.tight_layout()

    # Add a legend
    categories = data.get_firm_char().category2color.keys()
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=data.get_firm_char().category2color[cat],
        )
        for cat in categories
    ]
    plt.legend(
        handles,
        categories,
        loc="upper right",
        bbox_to_anchor=(1.25, 1),
    )

    plt.show()


plot_feature_importance(model, data)

# %%
# Feature importance plot based on the gradients as it is done in the paper


def plot_feature_importance(model, data, X_test):
    # Define a function to compute gradients
    @tf.function
    def get_gradients(inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = model(inputs)
        return tape.gradient(predictions, inputs)

    # Calculate gradients for the test data
    gradients = get_gradients(X_test)

    # Take the absolute mean of gradients across all time steps and stocks
    feature_importance = tf.reduce_mean(tf.abs(gradients), axis=[0, 1]).numpy()

    normalized_importance = feature_importance / np.sum(feature_importance)

    feature_names = data.get_feature_names()
    importance_dict = dict(zip(feature_names, normalized_importance))

    sorted_importance = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=False,
    )

    color_map = data.get_firm_char().getColorLabelMap()

    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        range(len(sorted_importance)),
        [imp for _, imp in sorted_importance],
        color=[color_map[feat] for feat, _ in sorted_importance],
    )
    plt.yticks(
        range(len(sorted_importance)),
        [feat for feat, _ in sorted_importance],
    )
    plt.ylabel("Stock Attributes")
    plt.xlabel("Importance")
    plt.title("Stock Attribute Importance Based on Gradients")
    plt.tight_layout()

    categories = data.get_firm_char().category2color.keys()
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=data.get_firm_char().category2color[cat],
        )
        for cat in categories
    ]
    plt.legend(
        handles,
        categories,
        loc="center right",
        bbox_to_anchor=(1, 0.5),
    )

    plt.show()


plot_feature_importance(model, data, df_train[:, :, 1:])
