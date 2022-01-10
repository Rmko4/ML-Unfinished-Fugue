from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from midi_duration import (MIDI_COMPACT, MIDI_COMPACT_SC,
                           midi_tones_file_to_midi_compact)
from midi_to_5d_vec import midi_to_5d_vec

if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    CHANNEL = 0  # Only generate model ready data for this channel
    OMIT_REST = True
    WINDOW_LENGTH = 10


def convert_to_windowed_input(midi_compact: MIDI_COMPACT_SC, window_length=10):
    pass


def convert_to_probabilty_duration_output(midi_compact: MIDI_COMPACT_SC, range: Tuple[int, int]):
    pass


#TODO: Typing
def convert_to_training_data(midi_compact: MIDI_COMPACT_SC, validation_split=0.0) -> MIDI_COMPACT:

    if validation_split >= 1.0 or validation_split < 0.0:
        raise ValueError(
            f"validation_split should be in range [0.0, 1.0),but is {validation_split}")

    symbol_len = len(midi_compact)

    sep = int((1 - validation_split) * symbol_len)

    train_raw = midi_compact[:sep]
    val_raw = midi_compact[sep:]
    convert_to_windowed_input(train_raw)
    pass

    # symbol_len = midi_notes.shape[0]
    # n_channels = midi_notes.shape[1]

    # compact = []

    # for channel in range(n_channels):
    #     voice = midi_notes[:, channel]

    #     compact_voice = []
    #     compact.append(compact_voice)

    #     prev_midi = voice[0]
    #     start_idx = 0
    #     for idx in range(1, symbol_len):
    #         midi_pitch = voice[idx]
    #         # Change to different tone
    #         if midi_pitch != prev_midi:
    #             if not omit_rest or prev_midi != 0:
    #                 duration = (idx - start_idx) / symbols_per_beat
    #                 pitch = prev_midi + modulation
    #                 compact_voice.append((pitch, duration))
    #             prev_midi = midi_pitch
    #             start_idx = idx

    # return compact


if __name__ == "__main__":
    midi_compact = midi_tones_file_to_midi_compact(
        FILENAME_F, omit_rest=OMIT_REST)
    in_d = midi_compact[CHANNEL]
    train, val = convert_to_training_data(in_d, validation_split=.2)
