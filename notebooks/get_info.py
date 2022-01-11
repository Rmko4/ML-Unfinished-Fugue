# %%
import pandas as pd
import numpy as np
from pandas import DataFrame
# %%
frequencies: DataFrame = pd.read_csv("F.txt", sep="\t", header=None)
frequencies = frequencies.to_numpy()

# %%
frequencies_masked = np.ma.masked_equal(frequencies, 0)
min_f = np.amin(frequencies_masked, 0)
max_f = np.amax(frequencies_masked, 0)

chroma_range = max_f - min_f + 1

print(f'min freq={min_f}')
print(f'max freq={max_f}')
print(f'chroma range={chroma_range}')
# %%
modulation = 8

min_f_mod = min_f + modulation
max_f_mod = max_f + modulation

print(f'min_f_mod={min_f_mod}')
print(f'max_f_mod={max_f_mod}')

# %%
