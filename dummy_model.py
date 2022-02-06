
import sys
import numpy as np
from pathlib import Path
import pandas as pd
from data_io.midi_file import (MODULATION, TEMPO, midi_compact_to_midi_file,
                               midi_tones_to_midi_file)
from postprocessing.postprocessing import PostProcessorMC
from data_io.model_data import (convert_midi_compact_sc_to_training_data,
                                convert_raw_to_training_data, load_data_raw)

class DummyOutputVectorEncoder():
    def __init__(self):
        self.n_channels = 4
        self.flatten = False
        self.note_min = [54,45,40,28]
        self.note_max = [76,71,62,54]


OUTPUT_PATH = Path("output_midi_files")
N_TIMESTEPS = 486
ove = DummyOutputVectorEncoder()
midi_raw = load_data_raw("F.txt")[:-16, :]

post_processor = PostProcessorMC(ove, midi_raw ,16)

predicted_sequence = []
y = []
for j in range(ove.n_channels):
    y.append([[1 / (ove.note_max[j]+1 - ove.note_min[j]) for x in range(ove.note_min[j], ove.note_max[j]+1)]])


for i in range(N_TIMESTEPS):
    predicted_sequence.append(post_processor.__call__(y,None))


pd.DataFrame(predicted_sequence).to_csv(
    "analyseData/dummy_model.txt", header=None, index=None, sep='\t')

output_file = OUTPUT_PATH / "dummy_model.mid"
midi_tones_to_midi_file(np.array(predicted_sequence), str(output_file),
                        tempo=TEMPO, modulation=MODULATION)

output_file = "output_midi_files/full_seq_plus_pred_dummy_model.mid"
full_sequence = midi_raw.copy()
song = np.concatenate((full_sequence, predicted_sequence))
midi_tones_to_midi_file(song, str(output_file), tempo=TEMPO, modulation=MODULATION)
