# %% [markdown]
# Generating a wav file from the midi file
# %%
import numpy as np
import pandas as pd
from IPython.display import Audio
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
from scipy.io.wavfile import write

filename_F = "F.txt"

# %%
frequencies: DataFrame = pd.read_csv(filename_F, sep="\t", header=None)
frequencies = frequencies.to_numpy()


# %%
def midi_to_freq(midi):
    return 440*2**((midi-69)/12)

# %%
def gen_wave(voice, f_s, symbol_dur):
    symbol_len = len(voice)
    T_s = 1 / f_s
    n_per_symbol = f_s * symbol_dur
    voice_sampled = np.zeros((int(symbol_len * n_per_symbol)))

    prev_midi = voice[0]
    start_idx = 0
    for idx, midi in enumerate(voice[1:]):
        # Change to different tone
        if midi != prev_midi:
            if prev_midi != 0:
                start_n = int(start_idx * n_per_symbol)
                end_n = int(idx * n_per_symbol)
                n_tone_len = end_n - start_n
                freq = midi_to_freq(prev_midi)

                # Digital frequency
                omega_hat = 2 * np.pi * freq * T_s

                n = np.linspace(0, n_tone_len, n_tone_len, endpoint=False)
                tone_wave = np.sin(omega_hat * n)
                voice_sampled[start_n:end_n] = tone_wave

            prev_midi = midi
            start_idx = idx
            
    return voice_sampled

# %%
bpm = 120  # Allegro - Played at 121
symbols_per_beat = 4
f_s = 44100
symbol_dur = 1 / (symbols_per_beat * bpm / 60)
symbol_len = frequencies.shape[0]
n_per_symbol = f_s * symbol_dur

wave = np.zeros((int(symbol_len * n_per_symbol)))
for voice_i in range(4):
    midi_vals = frequencies[:,voice_i]
    wave += gen_wave(midi_vals, f_s, symbol_dur)

wave /= 4

Audio(wave, rate=f_s)

# %%
write("voice_s.wav", rate=f_s, data=wave)
