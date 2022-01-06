# %% [markdown]

# %%
import numpy as np
import pandas as pd
from IPython.display import Audio
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame

filename_F = "F.txt"

# %%
frequencies: DataFrame = pd.read_csv(filename_F, sep="\t", header=None)
frequencies = frequencies.to_numpy()

# %%
voice_s = frequencies[:, 0]
# TODO: Make sure to remove the last measure
plt.plot(voice_s)


def midi_to_freq(midi):
    return 440*2**((midi-69)/12)

# %%
bpm = 120  # Allegro - Played at 121
symbols_per_beat = 4
f_s = 44.1e3

symbol_dur = 1 / (symbols_per_beat * bpm / 60)
T_s = 1 / f_s

n_len = len(voice_s)
n_per_symbol = f_s * symbol_dur
voice_s_sampled = np.zeros((n_len * n_per_symbol))
# %%
