from esn import ESN
from data_io.midi_duration import (MIDI_COMPACT_SC,
                                   midi_tones_file_to_midi_compact)
from data_io.midi_file import MODULATION, TEMPO, midi_compact_to_midi_file, midi_tones_to_midi_file
from data_io.model_data import convert_midi_compact_sc_to_training_data, convert_raw_to_training_data, load_data_raw
from data_io.vector_encoders import (InputVectorEncoderMC,
                                     InputVectorEncoderSC,
                                     OutputVectorEncoderMC,
                                     OutputVectorEncoderSC)


midi_raw = load_data_raw('F.txt')[:-16, :]
data, ove, ive = convert_raw_to_training_data(
    midi_raw, window_length=1, flatten_output=True)

u, y = data

esn = ESN(u.shape[1], y.shape[1], silent=False)
esn.fit(u, y, save_states=True)
