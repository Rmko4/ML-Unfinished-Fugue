from typing import Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pandas import DataFrame
from data_io.vector_encoders import *

from midi_duration import MIDI_COMPACT_SC, midi_tones_file_to_midi_compact

if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    CHANNEL = 0  # Only generate model ready data for this channel
    OMIT_REST = True


def convert_midi_compact_sc_to_training_data(midi_compact: MIDI_COMPACT_SC, window_length=10,
                                             validation_split=0.0) -> Tuple[np.ndarray, np.ndarray, OutputVectorEncoderSC, InputVectorEncoderSC]:

    if validation_split >= 1.0 or validation_split < 0.0:
        raise ValueError(
            f"validation_split should be in range [0.0, 1.0), but is {validation_split}")

    ove = OutputVectorEncoderSC(midi_compact)
    ive = InputVectorEncoderSC(midi_compact)

    def _encode(X: MIDI_COMPACT_SC):
        y = ove.transform(X)
        u = ive.transform(X)

        # Generates sliding windows view, with last window omitted, as this contains the last y.
        u_sw = sliding_window_view(u, window_length, axis=0)[:-1]
        # Flatten the second and third dimension
        u_sw = u_sw.reshape(u_sw.shape[0], -1)

        # Omit initial part of y, as this is part of the first window.
        y = y[window_length:]

        return u_sw, y

    symbol_len = len(midi_compact)

    sep = int((1 - validation_split) * symbol_len)

    train_raw = midi_compact[:sep]
    val_raw = midi_compact[sep:]

    train = _encode(train_raw)

    if len(val_raw) <= window_length + 1:
        val = None
    else:
        val = _encode(val_raw)

    return train, val, ove, ive


""" Returns ndarray with the first dimension for the time and the second dimension
    the frequency for each voice.
"""


def load_data_raw(file="F.txt") -> np.ndarray:
    midi_raw: DataFrame = pd.read_csv(file, sep="\t", header=None)
    midi_raw = midi_raw.to_numpy()
    return midi_raw


def convert_raw_to_training_data(midi_raw: np.ndarray, window_length=48,
                                 flatten_output=False) -> Tuple[np.ndarray, OutputVectorEncoderMC, InputVectorEncoderMC]:

    ove = OutputVectorEncoderMC(midi_raw, flatten_output)
    ive = InputVectorEncoderMC(midi_raw, True)

    def _encode(X: MIDI_COMPACT_SC):
        y = ove.transform(X)
        u = ive.transform(X)

        # Generates sliding windows view, with last window omitted, as this contains the last y.
        u_sw = sliding_window_view(u, window_length, axis=0)[:-1]
        # Flatten the second and third dimension
        u_sw = u_sw.reshape(u_sw.shape[0], -1)

        # Omit initial part of y, as this is part of the first window.
        if flatten_output:
            y = y[window_length:]
        else:
            y = [x[window_length:] for x in y]

        return u_sw, y

    train = _encode(midi_raw)

    return train, ove, ive


if __name__ == "__main__":
    midi_compact = midi_tones_file_to_midi_compact(
        FILENAME_F, omit_rest=OMIT_REST)
    # Only use channel CHANNEL; Last measure is omitted as this one is incomplete
    in_d = midi_compact[CHANNEL][:-16]
    train, val, ove, ive = convert_midi_compact_sc_to_training_data(
        in_d, validation_split=.2)

    in_d_raw = load_data_raw(FILENAME_F)[:, :]
    train_mc, tve_mc, ive_mc = convert_raw_to_training_data(in_d_raw, flatten_output=False)
