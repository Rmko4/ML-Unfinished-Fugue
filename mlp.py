from pathlib import Path
from typing import List

import numpy as np
import tensorflow.keras as keras

from data_io.midi_file import MODULATION, TEMPO, midi_tones_to_midi_file
from data_io.model_data import convert_raw_to_training_data, load_data_raw
from data_io.vector_encoders import InputVectorEncoderMC, OutputVectorEncoderMC
from model_extensions.predict_sequence import SequencePredictorMixin

if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    OUTPUT_PATH = Path("output_midi_files")
    OMIT_REST = True
    CHANNEL = 0
    WINDOW_LENGTH_MC = 48
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


class MultiLayerPerceptronMC(SequencePredictorMixin, keras.Model):
    def __init__(self, ive: OutputVectorEncoderMC,
                 ove: InputVectorEncoderMC, hidden_units, batch_size: int = None,
                 **kwargs):
        super().__init__(ive, ove, **kwargs)

        self.batch_size=batch_size

        self.hidden_0 = keras.layers.Dense(
            hidden_units,
            activation='sigmoid',
            name='hidden')

        self.outputs = []
        for channel in range(self.ove.n_channels):
            output_layer = keras.layers.Dense(
                self.ove.encoder_len[channel],
                activation='softmax',
                name=f'output_{channel}')
            self.outputs.append(output_layer)

    def call(self, inputs):
        x = self.hidden_0(inputs)
        y_pred = []
        for channel in range(self.ove.n_channels):
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
        if batch_size:
            self.batch_size = batch_size

        return super().fit(x, y, self.batch_size, epochs, verbose, callbacks, validation_split, **kwargs)


def apply_mlp_MC():
    midi_raw = load_data_raw(FILENAME_F)[:, :]
    train, ove, ive = convert_raw_to_training_data(
        midi_raw, window_length=WINDOW_LENGTH_MC, flatten_output=False)

    # Output should NOT be flattened for mlp.
    u, y = train

    mlp = MultiLayerPerceptronMC(ive, ove, 32)
    mlp.compile(optimizer='adam')

    callbacks = [keras.callbacks.EarlyStopping(patience=3)]

    mlp.fit(u, y, batch_size=32, epochs=100,
            validation_split=0.2, callbacks=callbacks)

    predicted_sequence = mlp.predict_sequence(
        midi_raw[-WINDOW_LENGTH_MC:], steps=N_NEW_SYMBOLS)

    full_sequence = np.concatenate((midi_raw, predicted_sequence), axis=0)

    output_file = OUTPUT_PATH / "pred_mlp_mc.mid"
    midi_tones_to_midi_file(predicted_sequence, str(output_file),
                            tempo=TEMPO, modulation=MODULATION)
    # Cross validate over batch size, number of units, window length
    # Keep max epochs same, but not too large.
    # Metric to determine best model?
    # Random grid search? 5 fold?


if __name__ == "__main__":
    apply_mlp_MC()
