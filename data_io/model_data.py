from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from midi_duration import (MIDI_COMPACT_SC,
                           midi_tones_file_to_midi_compact)
from vector_encoders import InputVectorEncoder, TeacherVectorEncoder

if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    CHANNEL = 0  # Only generate model ready data for this channel
    OMIT_REST = True


def convert_to_training_data(midi_compact: MIDI_COMPACT_SC, window_length=10,
                             validation_split=0.0) -> Tuple[np.ndarray, np.ndarray, TeacherVectorEncoder, InputVectorEncoder]:

    if validation_split >= 1.0 or validation_split < 0.0:
        raise ValueError(
            f"validation_split should be in range [0.0, 1.0), but is {validation_split}")

    tve = TeacherVectorEncoder(midi_compact)
    ive = InputVectorEncoder(midi_compact)

    def _encode(X: MIDI_COMPACT_SC):
        y = tve.transform(X)
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

    return train, val, tve, ive


if __name__ == "__main__":
    midi_compact = midi_tones_file_to_midi_compact(
        FILENAME_F, omit_rest=OMIT_REST)
    # Only use channel CHANNEL; Last measure is omitted as this one is incomplete
    in_d = midi_compact[CHANNEL][:-16]
    train, val, tve, ive = convert_to_training_data(in_d, validation_split=.2)
