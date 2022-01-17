from typing import List
import numpy as np
import tensorflow.keras as keras
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LinearRegression

from data_io.midi_file import MODULATION, TEMPO, midi_compact_to_midi_file
from data_io.model_data import (convert_midi_compact_sc_to_training_data,
                                convert_raw_to_training_data, load_data_raw)
from data_io.vector_encoders import (InputVectorEncoder, InputVectorEncoderSC,
                                     TeacherVectorEncoder,
                                     TeacherVectorEncoderSC)

if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    OMIT_REST = True
    CHANNEL = 0
    WINDOW_LENGTH = 10
    N_NEW_SYMBOLS = 160  # Roughly 20 seconds considering bpm 120 and 4 symbols per beat


class MetricPrintCallback(keras.callbacks.Callback):
    '''
    Due to the possibly many outputs by a keras model, the fit procedure might
    not be able to print metrics clearly.
    This class provides a printing callback for the training and validation loss.
    '''

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        print(f'Epoch: {epoch}, Train Loss: {loss}, Val Loss: {val_loss}')

    def on_epoch_begin(self, epoch, logs=None):
        print('-'*50)


class LinearRegressionSC(LinearRegression):
    def __init__(self, tve: TeacherVectorEncoderSC, ive: InputVectorEncoderSC,
                 fit_intercept=True, normalize=False,
                 copy_X=True, n_jobs=None, positive=False) -> None:
        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        self.tve = tve
        self.ive = ive

    def predict_sequence(self, X: np.ndarray, duration=64):
        mc_pred_seq = []
        u = self.ive.transform(X)
        time = 0
        while time < duration:
            # Create single window view
            u_sw = sliding_window_view(u, len(X), axis=0)
            u_sw = u_sw.reshape(u_sw.shape[0], -1)

            # Raw output
            y_pred = self.predict(u_sw)
            # Select note and duration on maximum likelihood
            out = self.tve.inv_transform_maximum_likelihood(y_pred)
            mc_pred = (int(out[0][0]), int(out[0][1]))

            # Add the prediction to the sequence
            mc_pred_seq.append(mc_pred)

            # Update window with new prediction
            new_u = self.ive.transform(out)
            u = np.concatenate((u[1:], new_u))

            # Update total duration with new note
            time += mc_pred[1]

        return mc_pred_seq


class MultiLayerPerceptronMC(keras.Model):
    def __init__(self, hidden_units, tve: TeacherVectorEncoder,
                 ive: InputVectorEncoder, **kwargs):
        super().__init__(**kwargs)
        self.tve = tve
        self.ive = ive

        self.hidden_0 = keras.layers.Dense(
            hidden_units,
            activation='sigmoid',
            name='hidden')

        self.outputs = []
        for channel in range(self.tve.n_channels):
            output_layer = keras.layers.Dense(
                self.tve.encoder_len[channel],
                activation='softmax',
                name=f'output_{channel}')
            self.outputs.append(output_layer)

    def call(self, inputs):
        x = self.hidden_0(inputs)
        y_pred = []
        for channel in range(self.tve.n_channels):
            out = self.outputs[channel](x)
            y_pred.append(out)

        return y_pred

    def compile(self, optimizer='adam', run_eagerly=None, **kwargs):
        loss = keras.losses.CategoricalCrossentropy()
        return super().compile(optimizer, loss, run_eagerly=run_eagerly, **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, callbacks: List = [],
            validation_split=0, **kwargs):
        verbose = 0
        callbacks.append([MetricPrintCallback()])
        return super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, **kwargs)

    def predict_sequence(self, X: np.ndarray, steps=64):
        pred_seq = []

        raise NotImplementedError()

def apply_mlp():
    midi_raw = load_data_raw(FILENAME_F)[:, :]
    # Only use channel CHANNEL; Last measure is omitted as this one is incomplete
    train, tve, ive = convert_raw_to_training_data(
        midi_raw, flatten_output=False)

    # Output should NOT be flattened for mlp.
    u, y = train

    mlp = MultiLayerPerceptronMC(32, tve, ive)
    mlp.compile(optimizer='adam')

    callbacks = [keras.callbacks.EarlyStopping(patience=3)]

    mlp.fit(u, y, batch_size=32, epochs=100,
            validation_split=0.2, callbacks=callbacks)

    # Cross validate over batch size, number of units, window length
    # Keep max epochs same, but not too large.
    # Metric to determine best model?
    # Random grid search? 5 fold?

    pass


if __name__ == "__main__":
    apply_mlp()
