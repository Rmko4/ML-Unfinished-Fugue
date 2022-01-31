from audioop import bias
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
u_train = u[:3045, :]
y_train = y[:3045, :]
u_val = u[3046:, :]
y_val = y[3046:, :]

esn = ESN(u.shape[1], y.shape[1], silent=False, reservoir_size=200, activation_func='tanh',
          W_in_scaling=0.5, W_scaling=0.2, W_fb_scaling=0.5, bias_scaling=0.2, spectral_radius=0.95,
          leaking_rate=0.95, washout_time=3, ive=ive, ove=ove)

esn.fit(u_train, y_train)
esn.validate(u_val, y_val)
# esn.determine_washout_time(u, y, 3, 10)
# esn.save_model()
# predicted_sequence = esn.predict_sequence(u, y, 100, ive, ove)

# output_file = "pred_esn_mc.mid"
# midi_tones_to_midi_file(predicted_sequence, str(output_file),
# tempo = TEMPO, modulation = MODULATION)
