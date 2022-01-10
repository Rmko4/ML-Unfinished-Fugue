from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

MIDI_COMPACT_SC = List[Tuple[int, float]]
MIDI_COMPACT = List[MIDI_COMPACT_SC]


if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    OMIT_REST = True


def midi_tones_to_midi_compact(midi_notes: np.ndarray, omit_rest=True, modulation=0, symbols_per_beat=4) -> MIDI_COMPACT:
    symbol_len = midi_notes.shape[0]
    n_channels = midi_notes.shape[1]

    compact = []

    for channel in range(n_channels):
        voice = midi_notes[:, channel]

        compact_voice = []
        compact.append(compact_voice)

        prev_midi = voice[0]
        start_idx = 0
        for idx in range(1, symbol_len):
            midi_pitch = voice[idx]
            # Change to different tone
            if midi_pitch != prev_midi:
                if not omit_rest or prev_midi != 0:
                    duration = (idx - start_idx) / symbols_per_beat
                    pitch = prev_midi + modulation
                    compact_voice.append((pitch, duration))
                prev_midi = midi_pitch
                start_idx = idx

    return compact


def midi_tones_file_to_midi_compact(in_file="F.txt", **kwargs) -> MIDI_COMPACT:
    frequencies: DataFrame = pd.read_csv(in_file, sep="\t", header=None)
    frequencies = frequencies.to_numpy()
    return midi_tones_to_midi_compact(frequencies, **kwargs)


if __name__ == "__main__":
    out = midi_tones_file_to_midi_compact(FILENAME_F, omit_rest=OMIT_REST)

    for channel in out:
        print(len(channel))
    # First 20 notes of the first channel
    print(out[0][0:20])
