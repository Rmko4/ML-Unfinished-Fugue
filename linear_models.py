from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from data_io.midi_duration import (MIDI_COMPACT_SC,
                                   midi_tones_file_to_midi_compact)
from data_io.midi_file import (MODULATION, TEMPO, midi_compact_to_midi_file,
                               midi_tones_to_midi_file)
from data_io.model_data import (convert_midi_compact_sc_to_training_data,
                                convert_raw_to_training_data, load_data_raw)
from data_io.vector_encoders import (InputVectorEncoderMC,
                                     InputVectorEncoderSC,
                                     OutputVectorEncoderMC,
                                     OutputVectorEncoderSC)
from model_extensions.predict_sequence import SequencePredictorMixin
from postprocessing.postprocessing import PostProcessorMC

if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    OUTPUT_PATH = Path("output_midi_files")
    OMIT_REST = True
    MEASURE_LEN = 16 # Lenght of a measure in symbols
    POST_PROCESS = True
    CHANNEL = 0
    ALPHA = 2.6
    WINDOW_LENGTH_SC = 10
    WINDOW_LENGTH_MC = 378
    N_NEW_SYMBOLS = 486  # Roughly 20 seconds considering bpm 120 and 4 symbols per beat


class LinearRegressionSC(LinearRegression):
    def __init__(self, ove: OutputVectorEncoderSC, ive: InputVectorEncoderSC,
                 fit_intercept=True, normalize=False,
                 copy_X=True, n_jobs=None, positive=False) -> None:
        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        self.ove = ove
        self.ive = ive

    def predict_sequence(self, X: np.ndarray, duration=64) -> MIDI_COMPACT_SC:
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
            out = self.ove.inv_transform_maximum_likelihood(y_pred)
            mc_pred = (int(out[0][0]), int(out[0][1]))

            # Add the prediction to the sequence
            mc_pred_seq.append(mc_pred)

            # Update window with new prediction
            new_u = self.ive.transform(out)
            u = np.concatenate((u[1:], new_u))

            # Update total duration with new note
            time += mc_pred[1]

        return mc_pred_seq


class RidgeRegressionMC(SequencePredictorMixin, Ridge):
    def __init__(self, ive: InputVectorEncoderMC, ove: OutputVectorEncoderMC,
                 alpha=1, **kwargs):
        super().__init__(ive, ove, alpha=alpha, **kwargs)


class LinearRegressionMC(SequencePredictorMixin, LinearRegression):
    def __init__(self, ive: InputVectorEncoderMC, ove: OutputVectorEncoderMC, **kwargs):
        super().__init__(ive, ove, **kwargs)


def apply_linear_regression_SC():
    midi_compact = midi_tones_file_to_midi_compact(
        FILENAME_F, omit_rest=OMIT_REST)
    # Only use channel CHANNEL; Last measure is omitted as this one is incomplete
    in_d = midi_compact[CHANNEL][:-16]
    data, _, ove, ive = convert_midi_compact_sc_to_training_data(
        in_d, validation_split=0.1, window_length=WINDOW_LENGTH_SC)

    u, y = data

    train_u, val_u, train_y, val_y = train_test_split(u, y, test_size=0.2)

    lreg = LinearRegressionSC(ove, ive)

    lreg.fit(train_u, train_y)
    print(lreg.score(val_u, val_y))

    predicted_sequence = lreg.predict_sequence(
        in_d[-WINDOW_LENGTH_SC:], duration=N_NEW_SYMBOLS)

    full_sequence = in_d.copy()
    full_sequence.extend(predicted_sequence)
    midi_compact_fs = [full_sequence]

    output_file = OUTPUT_PATH / "pred.mid"
    midi_compact_to_midi_file(
        midi_compact_fs, str(output_file), tempo=TEMPO, modulation=MODULATION)


def apply_linear_regression_MC():
    midi_raw = load_data_raw(FILENAME_F)[:-16, :]
    data, ove, ive = convert_raw_to_training_data(
        midi_raw, window_length=WINDOW_LENGTH_MC, flatten_output=True)
    # Training data must be flattened for linear regressor

    u, y = data

    train_u, val_u, train_y, val_y = train_test_split(u, y, test_size=0.2)

    lreg = LinearRegressionMC(ive, ove)

    lreg.fit(train_u, train_y)
    print(lreg.score(val_u, val_y))

   # Will instantiate a post_processor if POST_PROCESS
    post_processor = PostProcessorMC(ove, midi_raw, measure_length=MEASURE_LEN)
    post_processing_func = post_processor if POST_PROCESS else None

    predicted_sequence = lreg.predict_sequence(
        midi_raw[-WINDOW_LENGTH_MC:], steps=N_NEW_SYMBOLS,
        inv_transform_fn=post_processing_func)

    pred_y = lreg.predict(u)

    pd.DataFrame(predicted_sequence).to_csv(
        "postprocessing/probabilities.txt", header=None, index=None, sep='\t')
    # read = pd.read_csv("postprocessing/probabilities.csv",
    #                    header=None).to_numpy()

    full_sequence = np.concatenate((midi_raw, predicted_sequence), axis=0)

    output_file = OUTPUT_PATH / "pred_linear_mc.mid"
    midi_tones_to_midi_file(predicted_sequence, str(output_file),
                            tempo=TEMPO, modulation=MODULATION)


def apply_ridge_regression_MC():
    # Load the raw midi data and omit the last full measure of 16 symbols.
    midi_raw = load_data_raw(FILENAME_F)[:-16, :]

    # Training data must be flattened for linear regressor.
    data, ove, ive = convert_raw_to_training_data(
        midi_raw, window_length=WINDOW_LENGTH_MC, flatten_output=True)

    u, y = data

    ridge = RidgeRegressionMC(ive, ove, alpha=ALPHA)
    ridge.fit(u, y)

    # For splitting into training and validation
    # train_u, val_u, train_y, val_y = train_test_split(u, y, test_size=0)
    # ridge.fit(train_u, train_y)
    # print(ridge.score(val_u, val_y))

    # Will instantiate a post_processor if POST_PROCESS
    post_processor = PostProcessorMC(ove, midi_raw, measure_length=MEASURE_LEN)
    post_processing_func = post_processor if POST_PROCESS else None

    predicted_sequence = ridge.predict_sequence(
        midi_raw[-WINDOW_LENGTH_MC:], steps=N_NEW_SYMBOLS,
        inv_transform_fn=post_processing_func)

    pd.DataFrame(predicted_sequence).to_csv(
        "analyseData/probabilities.txt", header=None, index=None, sep='\t')
    # read = pd.read_csv("postprocessing/probabilities.csv",
    #                    header=None).to_numpy()

    full_sequence = np.concatenate((midi_raw, predicted_sequence), axis=0)

    output_file = OUTPUT_PATH / "pred_ridge_mc.mid"
    midi_tones_to_midi_file(predicted_sequence, str(output_file),
                            tempo=TEMPO, modulation=MODULATION)

    output_file = "output_midi_files/full_seq_plus_linear_regression.mid"
    full_sequence = midi_raw.copy()
    song = np.concatenate((full_sequence, predicted_sequence))
    midi_tones_to_midi_file(song, str(output_file), tempo=TEMPO, modulation=MODULATION)


if __name__ == "__main__":
    # apply_linear_regression_SC()
    # apply_linear_regression_MC()
    apply_ridge_regression_MC()
