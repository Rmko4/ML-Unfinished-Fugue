import numpy as np
import pandas as pd
from midiutil import MIDIFile
from pandas.core.frame import DataFrame

FILENAME_F = "F.txt"  # Requires tabbed delimited csv
MODULATION = 8  # Key change
TEMPO = 121  # BPM
VOLUME = 100  # Max 127
INSTRUMENTS = [2, 2, 2, 2]  # Should be integer [1, 128]
# Now all bright acoustic piano.
# Look on https://en.wikipedia.org/wiki/General_MIDI for an overview of programs.


def midi_tones_to_midi_file(midi_notes: np.ndarray, out_file="out.mid",
                            symbols_per_beat=4, tempo=120, volume=100,
                            modulation=0, instruments=None) -> None:
    track = 0
    # One track, defaults to format 1 (tempo track is created automatically)
    midi = MIDIFile(1)
    midi.addTempo(track, 0, tempo)

    symbol_len = midi_notes.shape[0]
    n_channels = midi_notes.shape[1]

    if instruments is None:
        instruments = np.zeros((n_channels))
    else:
        instruments = np.array(instruments)
        assert len(instruments) == n_channels
        assert ((0 <= instruments) & (instruments < 128)).all()
        instruments -= 1

    for channel in range(n_channels):
        # For different instrument
        midi.addProgramChange(track, channel, 0, instruments[channel])
        voice = midi_notes[:, channel]

        prev_midi = voice[0]
        start_idx = 0
        for idx in range(1, symbol_len):
            midi_pitch = voice[idx]
            # Change to different tone
            if midi_pitch != prev_midi:
                if prev_midi != 0:
                    duration = (idx - start_idx) / symbols_per_beat
                    time = start_idx / symbols_per_beat
                    pitch = prev_midi + modulation
                    midi.addNote(track, channel, pitch, time, duration, volume)

                prev_midi = midi_pitch
                start_idx = idx

    with open(out_file, "wb") as output_file:
        midi.writeFile(output_file)


def midi_tones_file_to_midi_file(in_file="F.txt", out_file="out.mid", **kwargs) -> None:
    frequencies: DataFrame = pd.read_csv(in_file, sep="\t", header=None)
    frequencies = frequencies.to_numpy()
    midi_tones_to_midi_file(frequencies, out_file, **kwargs)


if __name__ == "__main__":
    midi_tones_file_to_midi_file(
        FILENAME_F, tempo=TEMPO, volume=VOLUME, modulation=MODULATION, instruments=INSTRUMENTS)
