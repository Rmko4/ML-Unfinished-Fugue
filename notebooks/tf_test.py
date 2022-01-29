# %%
import tensorflow as tf
import tensorflow.keras as keras

keras.layers.Dense

class new_model(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)