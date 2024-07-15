from keras import layers as L
from keras.models import Model


def ffn(
    input_shape: tuple,
    output_shape: tuple,
    output_activation: str = "softmax",
    hidden_units: tuple = (64, 32),
    hidden_activation: str = "relu",
    dropout: float = 0.2,
) -> Model:
    inp = L.Input(input_shape)
    x = inp

    for units in hidden_units:
        x = L.Dense(units, activation=hidden_activation)(x)
        x = L.Dropout(dropout)(x)
        x = L.LayerNormalization()(x)

    out_units = 1
    for dim in output_shape:
        out_units *= dim

    out = L.Dense(out_units, activation=output_activation)(x)
    out = L.Reshape(output_shape)(out)

    model = Model(inp, out)
    return model
