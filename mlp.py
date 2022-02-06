from pathlib import Path
from typing import List

import numpy as np
import tensorflow.keras as keras

from data_io.midi_file import MODULATION, TEMPO, midi_tones_to_midi_file
from data_io.model_data import convert_raw_to_training_data, load_data_raw
from data_io.vector_encoders import InputVectorEncoderMC, OutputVectorEncoderMC
from model_extensions.predict_sequence import SequencePredictorMixin
from postprocessing.postprocessing import PostProcessorMC
import pandas as pd

if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    OUTPUT_PATH = Path("output_midi_files")
    MODEL_SAVE_PATH = Path("models/mlp")
    MEASURE_LEN = 16
    POST_PROCESS = True
    SAVE_MODEL = False
    OMIT_REST = True
    CHANNEL = 0
    WINDOW_LENGTH_MC = 31
    N_NEW_SYMBOLS = 486  # Roughly 20 seconds considering bpm 120 and 4 symbols per beat


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
    def __init__(self, ive: InputVectorEncoderMC,
                 ove: OutputVectorEncoderMC, hidden_units, batch_size: int = None,
                 l2: float = None, **kwargs):
        super().__init__(ive, ove, **kwargs)

        self.batch_size = batch_size
        kernel_regularizer = keras.regularizers.l2(l2) if l2 else None

        self.hidden_0 = keras.layers.Dense(
            hidden_units,
            activation='sigmoid',
            kernel_regularizer=kernel_regularizer,
            name='hidden')

        self.outputs = []
        for channel in range(self.ove.n_channels):
            output_layer = keras.layers.Dense(
                self.ove.encoder_len[channel],
                activation='softmax',
                kernel_regularizer=kernel_regularizer,
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
            validation_split=0, verbose=1, **kwargs):
        fwd_cb = callbacks.copy()
        if verbose == 1:  # Verbose 0 will not print intermediate input
            fwd_cb.append([MetricPrintCallback()])
        if batch_size:
            self.batch_size = batch_size

        return super().fit(x, y, self.batch_size, epochs, callbacks=fwd_cb,
                           validation_split=validation_split, verbose=0, **kwargs)


def apply_mlp_MC():
    midi_raw = load_data_raw(FILENAME_F)[:, :]
    train, ove, ive = convert_raw_to_training_data(
        midi_raw, window_length=WINDOW_LENGTH_MC, flatten_output=False)

    # Output should NOT be flattened for mlp.
    u, y = train

    mlp = MultiLayerPerceptronMC(ive, ove, hidden_units=410)
    mlp.compile(optimizer='adam')

    callbacks = [keras.callbacks.EarlyStopping(patience=3)]
    if SAVE_MODEL:
        callbacks.append(keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH))

    mlp.fit(u, y, batch_size=32, epochs=100,
            validation_split=0.2, callbacks=callbacks)

   # Will instantiate a post_processor if POST_PROCESS
    post_processor = PostProcessorMC(ove, midi_raw, measure_length=MEASURE_LEN)
    post_processing_func = post_processor if POST_PROCESS else None

    predicted_sequence = mlp.predict_sequence(
        midi_raw[-WINDOW_LENGTH_MC:], steps=N_NEW_SYMBOLS,
        inv_transform_fn=post_processing_func)

    pd.DataFrame(predicted_sequence).to_csv(
        "analyseData/mlp_with_postprocessing.txt", header=None, index=None, sep='\t')


    full_sequence = np.concatenate((midi_raw, predicted_sequence), axis=0)

    output_file = OUTPUT_PATH / "pred_mlp_mc.mid"
    midi_tones_to_midi_file(predicted_sequence, str(output_file),
                            tempo=TEMPO, modulation=MODULATION)

    output_file = "output_midi_files/full_seq_plus_pred_mlp_mc.mid"
    full_sequence = midi_raw.copy()
    song = np.concatenate((full_sequence, predicted_sequence))
    midi_tones_to_midi_file(song, str(output_file), tempo=TEMPO, modulation=MODULATION)


if __name__ == "__main__":
    apply_mlp_MC()
