# %%
import pandas as pd
import numpy as np
from pandas import DataFrame
# %%
frequencies: DataFrame = pd.read_csv("F.txt", sep="\t", header=None)
frequencies = frequencies.to_numpy()

#%%
frequencies_masked = np.ma.masked_equal(frequencies, 0)
min_f = np.amin(frequencies_masked, 0)
max_f = np.amax(frequencies_masked, 0)

chroma_range = max_f - min_f + 1

print(min_f)
print(max_f)
print(chroma_range)
# %%
modulation = 8

min_f_mod = min_f + modulation
max_f_mod = max_f + modulation

print(min_f_mod)
print(max_f_mod)

# %%
