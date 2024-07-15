import sys


def get_adam():
    """Get the right adam optimizer depending on the platform used."""

    if sys.platform == "darwin":
        try:
            from keras.optimizers.legacy import Adam
        except ImportError:
            from keras.optimizers import Adam
    else:
        from keras.optimizers import Adam

    return Adam
